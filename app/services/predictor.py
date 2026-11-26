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

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "Interaction": ("interaction/models/selected.joblib", INPUTS,
                    "Biểu diễn tương tác giữa thời lượng dùng AI, mục đích sử dụng và kỹ năng viết prompt."),
    "Polynomial": ("polynomial/models/selected_polynomial_pipeline.joblib", INPUTS,
                   "Dùng đặc trưng đa thức để biểu diễn quan hệ cong giữa đầu vào và GPA."),
    "Ridge": ("ridge/models/ridge_pipeline.joblib", NUM + BIN + CAT,
              "Thu nhỏ hệ số hồi quy bằng phạt L2. Pipeline này không dùng mức lo âu khi thi."),
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
    pipeline = joblib.load(ROOT / "outputs" / MODELS[name][0])
    pipeline.predict(dataset("train").iloc[:1][MODELS[name][1]])
    return pipeline


def predict(name, frame):
    return model(name).predict(frame.loc[:, MODELS[name][1]])


@st.cache_data
def validation(name):
    rows = dataset("validation")
    if name == "Ridge":
        run = json.loads((ROOT / "outputs/ridge/metrics.json").read_text())
        if run["config"]["refit_on_trval"] or run["n_fit"] != len(dataset("train")):
            raise ValueError("Ridge chưa xác minh được fit riêng trên train.")
    actual, predicted = rows[TARGET], predict(name, rows)
    scores = {"Thuật toán": name, "MAE": mean_absolute_error(actual, predicted),
              "RMSE": root_mean_squared_error(actual, predicted), "R²": r2_score(actual, predicted)}
    return scores, pd.DataFrame({"Thực tế": actual, "Dự đoán": predicted})
