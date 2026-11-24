# Quickstart dự kiến: THÂN Interaction Linear Regression

Các lệnh dưới đây là **kế hoạch nghiệm thu sau implementation**, chưa được xác nhận chạy ở bước plan. Chạy từ repository root. Không tải data qua mạng mặc định; cần chuẩn bị split/local CSV hoặc mount data. Joblib chỉ load artifact tự tạo/nguồn tin cậy.

## 1. Chuẩn bị dữ liệu

Đọc [data contract](contracts/model-io.md). Repo hiện có `data/assignments.csv`, `data/metadata.json`, ba CSV `data/splits/{train,val,test}.csv` và raw `data/raw/ai_student_impact_dataset (1).csv`. Cần xác nhận assignment dùng chung vì metadata ghi khác EDA gốc. Không tự tạo `frozen_split_ids.csv` mới khi assignment đã có; không overwrite dữ liệu gốc. Nếu đường dẫn khác, cấu hình theo README thực tế và dùng env `LAB_DATA_DIR`/`LAB_PROJECT_ROOT` khi implementation cung cấp. Trong mọi trường hợp xác minh integrity trước training.

## 2. Windows PowerShell, Python 3.11+

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest tests/test_features.py tests/test_leakage_and_splits.py tests/test_notebook_smoke.py tests/test_training.py tests/test_inference.py
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/interaction_linear_regression.ipynb --output executed_notebook.ipynb --output-dir outputs/interaction/manual-run
```

Cần tạo trước thư mục output nếu tool không tự tạo. Notebook phải ghi config/run ID `manual-run` tương ứng, hoặc README sau implementation phải dùng một lệnh runner đã định nghĩa để tránh notebook và nbconvert ghi hai run khác nhau. Trong Jupyter, mở notebook và chọn **Restart Kernel → Run All** từ môi trường root. Nếu chưa có notebook, dependencies hoặc split hợp lệ, lệnh sẽ không thành công; không báo đã chạy.

## 3. Kaggle / Colab

Đặt repo và pre-split data vào storage đã mount, mở terminal tại repo root (hoặc đổi sang project root từ cell setup, không hardcode đường dẫn cá nhân). Ví dụ shell trong notebook:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest tests/test_features.py tests/test_leakage_and_splits.py tests/test_notebook_smoke.py
mkdir -p outputs/interaction/manual-run
python -m jupyter nbconvert --to notebook --execute notebooks/interaction_linear_regression.ipynb --output executed_notebook.ipynb --output-dir outputs/interaction/manual-run
```

Lệnh cài dependency có thể cần mạng hoặc packages đã cache; phân tích/runtime không tự tải dataset. Truyền path mount qua config/env theo README. ipywidgets optional; dùng fallback nếu không được frontend hỗ trợ.

## 4. Kiểm tra nghiệm thu

1. Test fixtures xác nhận frozen split reuse, no leakage/fold-only fit, categorical, 39/41/45/53/55 với đủ levels, selection/tie, test lock, inference và notebook smoke; fixture không chứng minh kết quả thật.
2. Với data thật, kiểm `artifacts/data_manifest.json`, `outputs/interaction/<run_id>/metrics/model_lock.json`, comparison/CV, final test JSON, `models/`, ≥10 figures, notebook đã execute và provenance hash theo [contract](contracts/model-io.md). Đối chiếu model lock trước test và `fit_scope=train` của final model; validation metrics cũng phải dùng model train-only.
3. Nhập một raw profile hợp lệ qua `predict_student(raw_dict, model)` và chạy what-if AI Hours; thử category lạ, thiếu field, no-D/pre-exam và đường dẫn không tồn tại. Output phải có cảnh báo hoặc lỗi cụ thể.
4. Không có data thật: ghi `real training: not executed — missing configured dataset/split`, không tạo giả metrics. Kiểm định nghiệm thu chỉ đạt khi có bằng chứng real run và review; split/governance phải được thống nhất trước implementation.
