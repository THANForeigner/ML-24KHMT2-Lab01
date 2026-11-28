"""Read-only inference. Never fit models or consume the held-out test split."""
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lab01_ml.models.piecewise_compat import load_piecewise
from lab01_ml.features.engineering import INPUTS
from lab01_ml.models.ridge import NUM, BIN, CAT, TARGET
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
SIMPLE_FEATURES = dict(zip(
    ["M01", "M02", "M03", "M04", "M05"],
    ["Weekly_GenAI_Hours", "Traditional_Study_Hours", "Pre_Semester_GPA",
     "Tool_Diversity", "Perceived_AI_Dependency"],
))
PATHS = {
    "Simple": "simple/runs/20261125_105509_d606e49/models/M03_raw_ols.joblib",
    "Interaction": "interaction/models/selected.joblib",
    "Piecewise": "piecewise/models/s6_train_only.joblib",
    "Polynomial": "polynomial/models/selected_polynomial_pipeline.joblib",
    "Ridge": "ridge/models/ridge_pipeline.joblib",
    "Elastic Net": "elastic_net/models/selected_elastic_net_pipeline.joblib",
}
LABELS = [
    "GPA trước học kỳ", "Giờ tự học truyền thống", "Năm học", "Nhóm ngành",
    "Giờ dùng AI mỗi tuần", "Mục đích dùng AI", "Kỹ năng viết prompt", "Số công cụ",
    "Có thuê bao trả phí", "Quy định của trường", "Mức phụ thuộc AI", "Mức lo âu khi thi",
]
DESCRIPTIONS = [
    "Một biến đầu vào · M01–M05", "Tương tác giữa các đặc trưng",
    "Tuyến tính từng đoạn", "Đặc trưng đa thức", "Điều chuẩn L2", "Điều chuẩn L1 + L2",
]
CONTINUOUS = {"Pre_Semester_GPA", "Traditional_Study_Hours", "Weekly_GenAI_Hours"}
CURVE_FIELDS = ["Weekly_GenAI_Hours", "Pre_Semester_GPA", "Traditional_Study_Hours",
                "Tool_Diversity", "Perceived_AI_Dependency"]


@lru_cache
def training():
    path = ROOT / "data/train/dataset.csv"
    manifest = json.loads((ROOT / "data/metadata.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["dataset_sha256"]["train/dataset.csv"]:
        raise ValueError("Dữ liệu train không khớp SHA-256.")
    return pd.read_csv(path, usecols=INPUTS)


@lru_cache
def validation_rows():
    path = ROOT / "data/validation/dataset.csv"
    manifest = json.loads((ROOT / "data/metadata.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["dataset_sha256"]["validation/dataset.csv"]:
        raise ValueError("Dữ liệu validation không khớp SHA-256.")
    return pd.read_csv(path, usecols=INPUTS + [TARGET])


def fields_for(name, variant):
    if name == "Simple":
        return [SIMPLE_FEATURES[variant]]
    return NUM + BIN + CAT if name == "Ridge" else INPUTS


@lru_cache
def schema():
    rows = training()
    result = []
    for i, (field, label) in enumerate(zip(INPUTS, LABELS)):
        values = rows[field].dropna()
        item = {"id": field, "label": label, "group": 0 if i < 4 else 1 if i < 9 else 2}
        if field in CONTINUOUS:
            item.update(kind="number", min=float(values.min()), max=float(values.max()), step=.001)
        else:
            item.update(kind="select", options=sorted(values.unique().tolist()))
        result.append(item)
    return result


@lru_cache
def estimator(name, variant="M03"):
    relative = PATHS[name]
    if name == "Simple":
        relative = f"simple/runs/20261125_105509_d606e49/models/{variant}_raw_ols.joblib"
    path = ROOT / "outputs" / relative
    result = load_piecewise(path) if name == "Piecewise" else joblib.load(path)
    if name == "Simple":
        if (result["model_id"] != variant or result["feature"] != SIMPLE_FEATURES[variant]
                or result["clipping"] or result["prediction_policy"] != "raw_ols"):
            raise ValueError("Artifact Simple không đúng chính sách raw OLS.")
    if name == "Ridge":
        # Match the approved Streamlit policy, without modifying the artifact.
        result.named_steps["model"].upper = None
    return result


def infer(name, variant, rows):
    model = estimator(name, variant)
    if name == "Simple":
        values = model["estimator"].predict(rows[[SIMPLE_FEATURES[variant]]])
    else:
        frame = rows.loc[:, fields_for(name, variant)].copy()
        if name == "Piecewise":
            config = json.loads((ROOT / "outputs/piecewise/metrics.json").read_text())
            for field, levels in [("Year_of_Study", config["year_levels"]),
                                  ("Institutional_Policy", config["policy_levels"])]:
                frame[field] = frame[field].map(levels)
                if frame[field].isna().any():
                    raise ValueError(f"Không nhận dạng {field}.")
        values = model.predict(frame)
    if not np.isfinite(values).all():
        raise ValueError("Mô hình trả về kết quả không hữu hạn.")
    return values


def sample(index):
    rows = training().dropna()
    return rows.iloc[index % len(rows)].to_dict()


def validate_profile(profile):
    result = {}
    for field in schema():
        key = field["id"]
        value = profile.get(key)
        if field["kind"] == "number":
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or not field["min"] <= value <= field["max"]):
                raise ValueError(f"{field['label']}: nhập số trong miền {field['min']}–{field['max']}.")
        else:
            if key == "Paid_Subscription":
                valid = type(value) is bool
            else:
                valid = value is not None and value in field["options"] and not isinstance(value, bool)
            if not valid:
                raise ValueError(f"{field['label']}: lựa chọn không hợp lệ.")
        result[key] = value
    return result


def predict(name, variant, profile, variable):
    frame = pd.DataFrame([profile])
    gpa = float(infer(name, variant, frame)[0])
    domain = training()[variable].dropna()
    grid = (np.linspace(domain.min(), domain.max(), 61) if variable in CONTINUOUS
            else np.sort(domain.unique()))
    grid = np.unique(np.r_[grid, profile[variable]])
    curve_frame = pd.DataFrame([profile] * len(grid)).assign(**{variable: grid})
    values = infer(name, variant, curve_frame)
    return {
        "model": name, "variant": variant if name == "Simple" else None,
        "profile": profile, "variable": variable, "gpa": gpa,
        "out_of_range": not 0 <= gpa <= 4,
        "curve": [{"x": float(x), "gpa": float(y)} for x, y in zip(grid, values)],
    }


def compare(profile, variant):
    frame = pd.DataFrame([profile])
    rows = []
    for name in PATHS:
        try:
            gpa = float(infer(name, variant, frame)[0])
            rows.append({"model": name, "gpa": gpa, "out_of_range": not 0 <= gpa <= 4, "error": None})
        except (OSError, ValueError, KeyError, AttributeError):
            logging.getLogger(__name__).exception("Comparison inference failed: %s", name)
            rows.append({"model": name, "gpa": None, "out_of_range": False,
                         "error": "Không tải được mô hình. Kiểm tra artifact và môi trường Python."})
    return {"profile": profile, "variant": variant, "rows": rows}


@lru_cache
def validation_result(name):
    if name == "Simple" or name not in PATHS:
        raise ValueError("Mô hình không đủ điều kiện đánh giá validation.")
    if name == "Ridge":
        config = json.loads((ROOT / "outputs/ridge/metrics.json").read_text())
        if config["config"]["refit_on_trval"] or config["n_fit"] != len(training()):
            raise ValueError("Chưa xác minh Ridge chỉ fit trên train.")
    rows = validation_rows()
    actual = rows[TARGET]
    predicted = infer(name, "M03", rows)
    metrics = {
        "model": name, "eligible": True, "reason": None,
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }
    return metrics, pd.DataFrame({"actual": actual, "predicted": predicted})


def validation_summary():
    metrics = []
    for name in PATHS:
        if name == "Simple":
            metrics.append({"model": name, "eligible": False, "mae": None, "rmse": None, "r2": None,
                            "reason": "Artifact đã fit trên train + validation; không chấm lại trên validation."})
            continue
        try:
            metrics.append(validation_result(name)[0])
        except (OSError, ValueError, KeyError, AttributeError):
            logging.getLogger(__name__).exception("Validation unavailable: %s", name)
            metrics.append({"model": name, "eligible": False, "mae": None, "rmse": None, "r2": None,
                            "reason": "Không thể xác minh kết quả validation. Kiểm tra artifact."})
    return {"n_rows": len(validation_rows()), "metrics": metrics}


def agreement(name):
    _, points = validation_result(name)
    lo, hi = float(points.min().min()), float(points.max().max())
    return {
        "model": name, "n_rows": len(points),
        "domain": [min(0, lo), max(4, hi)],
        "points": points.sample(min(800, len(points)), random_state=42).to_dict(orient="records"),
    }
