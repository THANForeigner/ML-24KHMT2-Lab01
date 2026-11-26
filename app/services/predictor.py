"""Inference with the selected, repository-owned lab pipelines."""
from pathlib import Path
import hashlib
import json

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from lab01_ml.features.engineering import INPUTS
from lab01_ml.models.ridge import NUM, BIN, CAT, TARGET
from services.piecewise_compat import load_piecewise

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "Simple": ("simple/runs/20261125_105509_d606e49/models/M03_raw_ols.joblib",
               ["Pre_Semester_GPA"],
               "Hồi quy tuyến tính đơn theo GPA trước học kỳ; dùng bản M03 không giới hạn (no clipping)."),
    "Interaction": ("interaction/models/selected.joblib", INPUTS,
                    "Biểu diễn tương tác giữa thời lượng dùng AI, mục đích sử dụng và kỹ năng viết prompt."),
    "Piecewise": ("piecewise/models/s6_train_only.joblib", INPUTS,
                  "Hồi quy tuyến tính từng đoạn S6 với các điểm gãy học từ train; không cắt dự đoán."),
    "Polynomial": ("polynomial/models/selected_polynomial_pipeline.joblib", INPUTS,
                   "Dùng đặc trưng đa thức để biểu diễn quan hệ cong giữa đầu vào và GPA."),
    "Ridge": ("ridge/models/ridge_pipeline.joblib", NUM + BIN + CAT,
               "Thu nhỏ hệ số bằng phạt L2. Dùng estimator đã fit, không cắt dự đoán ở GPA 4; không dùng mức lo âu khi thi."),
    "Elastic Net": ("elastic_net/models/selected_elastic_net_pipeline.joblib", INPUTS,
                    "Điều chỉnh mức phạt L1/L2 để thu nhỏ hệ số và chọn đặc trưng."),
}


@st.cache_data
def dataset(split):
    if split not in {"train", "validation"}:
        raise ValueError("Chỉ hỗ trợ train và validation")
    filename = f"{split}/dataset.csv"
    path = ROOT / "data" / filename
    metadata = json.loads((ROOT / "data/metadata.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["dataset_sha256"][filename]:
        raise ValueError(f"Sai mã SHA-256: {filename}")
    return pd.read_csv(path, usecols=INPUTS + ([TARGET] if split == "validation" else []))


@st.cache_resource
def model(name):
    path = ROOT / "outputs" / MODELS[name][0]
    pipeline = load_piecewise(path) if name == "Piecewise" else joblib.load(path)
    if name == "Simple":
        if (pipeline["model_id"] != "M03" or pipeline["clipping"]
                or pipeline["prediction_policy"] != "raw_ols"):
            raise ValueError("Simple: cần đúng model M03 no clipping.")
    else:
        if name == "Ridge":
            # This changes only the in-memory wrapper; fitted preprocessing and
            # Ridge coefficients stay untouched, and the joblib remains intact.
            pipeline.named_steps["model"].upper = None
        predict_with_model(name, pipeline, dataset("train").iloc[:1])
    return pipeline


def predict_with_model(name, pipeline, frame):
    if name == "Simple":
        return pipeline["estimator"].predict(frame[[pipeline["feature"]]])
    raw = frame.loc[:, MODELS[name][1]]
    if name == "Piecewise":
        config = json.loads((ROOT / "outputs/piecewise/metrics.json").read_text())
        raw = raw.copy()
        for field, levels in [("Year_of_Study", config["year_levels"]),
                              ("Institutional_Policy", config["policy_levels"])]:
            converted = raw[field].map(levels)
            if converted.isna().any():
                raise ValueError(f"Piecewise: giá trị {field} không hợp lệ.")
            raw[field] = converted
    return pipeline.predict(raw)


def predict(name, frame):
    return predict_with_model(name, model(name), frame)


@st.cache_data
def validation(name):
    if name == "Simple":
        raise ValueError("Artifact Simple đã fit trên train + validation; không chấm lại trên validation.")
    rows = dataset("validation")
    if name == "Ridge":
        run = json.loads((ROOT / "outputs/ridge/metrics.json").read_text())
        if run["config"]["refit_on_trval"] or run["n_fit"] != len(dataset("train")):
            raise ValueError("Ridge chưa xác minh được fit riêng trên train.")
    actual, predicted = rows[TARGET], predict(name, rows)
    scores = {"Thuật toán": name, "MAE": mean_absolute_error(actual, predicted),
              "RMSE": root_mean_squared_error(actual, predicted), "R²": r2_score(actual, predicted)}
    return scores, pd.DataFrame({"Thực tế": actual, "Dự đoán": predicted})
