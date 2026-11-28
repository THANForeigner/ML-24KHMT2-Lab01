"""The HTTP boundary must use real saved estimators, never preview constants."""
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from backend.api import app
    return TestClient(app)


@pytest.fixture
def profile():
    return pd.read_csv("data/train/dataset.csv").dropna().iloc[0].to_dict()


def test_bootstrap_exposes_six_models_and_twelve_editable_inputs(client):
    response = client.get("/api/bootstrap")
    assert response.status_code == 200
    data = response.json()
    assert [model["id"] for model in data["models"]] == [
        "Simple", "Interaction", "Piecewise", "Polynomial", "Ridge", "Elastic Net"
    ]
    assert len(data["fields"]) == 12
    assert len(data["profile"]) == 12
    assert "Graduate" in next(f for f in data["fields"] if f["id"] == "Year_of_Study")["options"]


@pytest.mark.parametrize("model,expected", [
    ("Interaction", 2.699), ("Ridge", 2.648),
    ("Polynomial", 2.562), ("Elastic Net", 2.685),
])
def test_predict_uses_real_models_and_returns_consistent_curve(client, profile, model, expected):
    response = client.post("/api/predict", json={
        "model": model, "profile": profile, "variable": "Weekly_GenAI_Hours",
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["gpa"] == pytest.approx(expected, abs=.0005)
    current = next(p for p in result["curve"] if p["x"] == profile["Weekly_GenAI_Hours"])
    assert current["gpa"] == pytest.approx(result["gpa"])
    assert result["model"] == model
    assert result["profile"]["Pre_Semester_GPA"] == profile["Pre_Semester_GPA"]


@pytest.mark.parametrize("variant,field", [
    ("M01", "Weekly_GenAI_Hours"), ("M02", "Traditional_Study_Hours"),
    ("M03", "Pre_Semester_GPA"), ("M04", "Tool_Diversity"),
    ("M05", "Perceived_AI_Dependency"),
])
def test_simple_variants_use_correct_saved_estimator(client, profile, variant, field):
    import joblib
    bundle = joblib.load(f"outputs/simple/runs/20261125_105509_d606e49/models/{variant}_raw_ols.joblib")
    expected = bundle["estimator"].predict(pd.DataFrame([profile])[[field]])[0]
    response = client.post("/api/predict", json={
        "model": "Simple", "variant": variant, "variable": field, "profile": profile,
    })
    assert response.status_code == 200, response.text
    assert response.json()["gpa"] == pytest.approx(expected)


def test_piecewise_loads_notebook_artifact(client, profile):
    response = client.post("/api/predict", json={
        "model": "Piecewise", "profile": profile, "variable": "Pre_Semester_GPA",
    })
    assert response.status_code == 200, response.text
    assert math.isfinite(response.json()["gpa"])


def test_ridge_preserves_prediction_above_four(client):
    profile = pd.read_csv("data/validation/dataset.csv").iloc[0].to_dict()
    response = client.post("/api/predict", json={
        "model": "Ridge", "profile": profile, "variable": "Pre_Semester_GPA",
    })
    assert response.status_code == 200, response.text
    assert response.json()["gpa"] > 4
    assert response.json()["out_of_range"] is True


@pytest.mark.parametrize("change", [
    {"model": "../../model"},
    {"variant": "M99"},
    {"profile": {}},
    {"profile": {"Pre_Semester_GPA": "NaN"}},
    {"variable": "Anxiety_Level_During_Exams"},
])
def test_bad_requests_do_not_reach_inference(client, profile, change):
    payload = {"model": "Simple", "variant": "M03", "profile": profile,
               "variable": "Pre_Semester_GPA"}
    payload.update(change)
    assert client.post("/api/predict", json=payload).status_code == 422


def test_category_and_numeric_domains_are_validated(client, profile):
    profile["Year_of_Study"] = "not-a-year"
    assert client.post("/api/predict", json={"model": "Ridge", "profile": profile}).status_code == 422
    profile["Year_of_Study"] = "Junior"
    profile["Perceived_AI_Dependency"] = 11
    assert client.post("/api/predict", json={"model": "Ridge", "profile": profile}).status_code == 422


def test_sample_is_complete_and_never_uses_test_split(client):
    response = client.get("/api/sample?index=2")
    assert response.status_code == 200
    assert len(response.json()["profile"]) == 12
    assert response.json()["profile"] != client.get("/api/sample?index=3").json()["profile"]


def test_comparison_predicts_all_six_from_one_submitted_profile(client, profile):
    response = client.post("/api/compare", json={"profile": profile, "variant": "M03"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["variant"] == "M03"
    assert result["profile"]["Pre_Semester_GPA"] == profile["Pre_Semester_GPA"]
    assert [row["model"] for row in result["rows"]] == [
        "Simple", "Interaction", "Piecewise", "Polynomial", "Ridge", "Elastic Net",
    ]
    assert [row["gpa"] for row in result["rows"]] == pytest.approx([
        2.650811, 2.698924, 2.644931, 2.562398, 2.647915, 2.684647,
    ], abs=.000001)


def test_comparison_simple_variant_matches_single_prediction(client, profile):
    result = client.post("/api/compare", json={"profile": profile, "variant": "M04"}).json()
    single = client.post("/api/predict", json={
        "profile": profile, "model": "Simple", "variant": "M04",
    }).json()
    assert result["rows"][0]["gpa"] == single["gpa"]


@pytest.mark.parametrize("payload", [{"profile": {}}, {"profile": {}, "variant": "M99"}])
def test_comparison_rejects_invalid_requests(client, payload):
    assert client.post("/api/compare", json=payload).status_code == 422


def test_validation_scores_exclude_simple_and_use_all_rows(client):
    response = client.get("/api/validation")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["n_rows"] == 7500
    rows = {row["model"]: row for row in result["metrics"]}
    assert set(rows) == {"Simple", "Interaction", "Piecewise", "Polynomial", "Ridge", "Elastic Net"}
    assert rows["Simple"]["eligible"] is False
    assert rows["Simple"]["rmse"] is None
    assert "train + validation" in rows["Simple"]["reason"]
    assert rows["Piecewise"]["rmse"] == pytest.approx(.14671197266237843)
    assert rows["Ridge"]["rmse"] == pytest.approx(.16076058372183918)
    assert rows["Interaction"]["r2"] == pytest.approx(.9082336734725989)


def test_agreement_returns_real_deterministic_sample_with_uncut_domain(client):
    response = client.get("/api/agreement?model=Ridge")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["n_rows"] == 7500
    assert len(result["points"]) == 800
    assert result == client.get("/api/agreement?model=Ridge").json()
    assert result["domain"][1] > 4
    # Pandas sample(random_state=42) picks this validation row first.
    rows = pd.read_csv("data/validation/dataset.csv")
    row = rows.sample(800, random_state=42).iloc[0]
    assert result["points"][0]["actual"] == row["Post_Semester_GPA"]
    expected = client.post("/api/predict", json={"profile": row.to_dict(), "model": "Ridge"}).json()
    assert result["points"][0]["predicted"] == pytest.approx(expected["gpa"])


@pytest.mark.parametrize("model", ["Simple", "unknown", "../../artifact"])
def test_agreement_rejects_ineligible_or_unknown_models(client, model):
    assert client.get("/api/agreement", params={"model": model}).status_code == 422


def test_one_missing_artifact_does_not_hide_other_comparison_results(client, profile, monkeypatch):
    from backend import service
    original = service.infer

    def infer_with_unavailable_piecewise(name, variant, frame):
        if name == "Piecewise":
            raise FileNotFoundError("fixture: missing artifact")
        return original(name, variant, frame)

    monkeypatch.setattr(service, "infer", infer_with_unavailable_piecewise)
    result = client.post("/api/compare", json={"profile": profile}).json()
    assert len(result["rows"]) == 6
    assert result["rows"][2]["gpa"] is None
    assert result["rows"][2]["error"]
    assert result["rows"][1]["gpa"] == pytest.approx(2.698924, abs=.000001)
