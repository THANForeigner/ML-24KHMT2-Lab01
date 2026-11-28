"""Run from the repository root: uvicorn backend.api:app --reload."""
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from backend import service

app = FastAPI(title="LAB / 01 · GPA inference", version="1.0.0")
log = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    variant: str = "M03"
    profile: dict[str, Any]
    variable: str | None = None


class ComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    variant: str = "M03"
    profile: dict[str, Any]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/bootstrap")
def bootstrap():
    return {
        "models": [
            {"id": name, "description": description,
             "fields": service.fields_for(name, "M03"),
             "available": (service.ROOT / "outputs" / path).is_file()}
            for (name, path), description in zip(service.PATHS.items(), service.DESCRIPTIONS)
        ],
        "fields": service.schema(), "profile": service.sample(0),
        "simple_variants": service.SIMPLE_FEATURES,
        "validation_count": len(service.validation_rows()),
    }


@app.get("/api/sample")
def sample(index: int = Query(0, ge=0)):
    return {"profile": service.sample(index)}


@app.post("/api/predict")
def predict(request: PredictionRequest):
    try:
        if request.model not in service.PATHS:
            raise ValueError("Mô hình không hợp lệ.")
        if request.variant not in service.SIMPLE_FEATURES:
            raise ValueError("Biến thể Simple không hợp lệ.")
        profile = service.validate_profile(request.profile)
        allowed = [field for field in service.CURVE_FIELDS
                   if field in service.fields_for(request.model, request.variant)]
        variable = request.variable or allowed[0]
        if variable not in allowed:
            raise ValueError("Biến khảo sát không được mô hình sử dụng.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return service.predict(request.model, request.variant, profile, variable)
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        log.exception("Saved estimator inference failed")
        raise HTTPException(status_code=503, detail="Không tải được mô hình. Kiểm tra artifacts và môi trường Python.") from exc


@app.post("/api/compare")
def compare(request: ComparisonRequest):
    try:
        if request.variant not in service.SIMPLE_FEATURES:
            raise ValueError("Biến thể Simple không hợp lệ.")
        profile = service.validate_profile(request.profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return service.compare(profile, request.variant)


@app.get("/api/validation")
def validation():
    try:
        return service.validation_summary()
    except (OSError, ValueError, KeyError) as exc:
        log.exception("Validation dataset unavailable")
        raise HTTPException(status_code=503, detail="Không tải được dữ liệu validation đã xác minh.") from exc


@app.get("/api/agreement")
def agreement(model: str = Query("Interaction")):
    if model == "Simple" or model not in service.PATHS:
        raise HTTPException(status_code=422, detail="Mô hình không đủ điều kiện đánh giá validation.")
    try:
        return service.agreement(model)
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        log.exception("Agreement data unavailable")
        raise HTTPException(status_code=503, detail="Không tải được biểu đồ validation cho mô hình này.") from exc


# Production: build the React app first. API and UI then share one origin.
dist = Path(__file__).resolve().parents[1] / "frontend/dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
