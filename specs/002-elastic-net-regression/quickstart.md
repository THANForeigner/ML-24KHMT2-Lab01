# Quickstart: Elastic Net Regression — Conditional GPA Prediction

Hướng dẫn kiểm tra môi trường, thực thi pipeline huấn luyện, chạy unit tests và tái lập kết quả nghiệm thu cho mô hình **Elastic Net Regression**.

---

## 1. Yêu cầu Môi trường & Dữ liệu

### Môi trường
- Python $\ge$ 3.11.
- Môi trường ảo `.venv` hoặc môi trường chuẩn trên Kaggle / Colab (chỉ cần CPU).

### Dữ liệu đóng băng (Frozen Splits)
Kiểm tra sự tồn tại của các file dữ liệu trong thư mục gốc của repository:
- `data/assignments.csv`
- `data/metadata.json`
- `data/train/dataset.csv` (hoặc `data/splits/train.csv`) — 35.000 dòng
- `data/validation/dataset.csv` (hoặc `data/splits/val.csv`) — 7.500 dòng
- `data/test/dataset.csv` (hoặc `data/splits/test.csv`) — 7.500 dòng (chỉ kiểm tra ID, không nạp target)

---

## 2. Hướng dẫn Chạy trên Linux / macOS / WSL

```bash
# 1. Kích hoạt môi trường ảo
source .venv/bin/activate

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
pip install -e .

# 3. Chạy toàn bộ Unit Tests kiểm tra tính toàn vẹn
pytest tests/ -q

# 4. Thực thi kiểm thử tính năng Elastic Net cụ thể
python -m unittest tests/test_features.py
python -m unittest tests/test_inference.py
python -m unittest tests/test_leakage_and_splits.py

# 5. Chạy notebook trong clean kernel bằng nbconvert/nbclient
jupyter nbconvert --to notebook --execute notebooks/elastic_net_regression.ipynb \
    --output executed_notebook.ipynb \
    --output-dir outputs/elastic_net/
```

---

## 3. Hướng dẫn Chạy trên Windows (PowerShell)

```powershell
# 1. Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt thư viện
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .

# 3. Chạy test suite
.\.venv\Scripts\python.exe -m pytest tests/ -q

# 4. Thực thi notebook
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/elastic_net_regression.ipynb `
    --output executed_notebook.ipynb `
    --output-dir outputs/elastic_net/
```

---

## 4. Hướng dẫn Chạy trên Kaggle / Google Colab

1. Tải repository hoặc clone vào workspace.
2. Đảm bảo dữ liệu `data/` được gắn đúng đường dẫn thư mục gốc.
3. Chạy cell cài đặt:
   ```python
   !pip install -r requirements.txt
   !pip install -e .
   ```
4. Mở trực tiếp notebook `notebooks/elastic_net_regression.ipynb`, chọn kernel Python 3 và bấm **Restart Kernel $\to$ Run All Cells**.

---

## 5. Tiêu chí Kiểm tra Nghiệm thu (Verification Checklist)

Sau khi hoàn tất quá trình thực thi, kiểm tra các điều kiện sau:

1. **Bảo vệ tập Holdout**: Trạng thái đánh giá tập test phải là `final test: not executed`. Target của test set không xuất hiện trong bất kỳ bảng điểm hay đồ thị nào.
2. **Số chiều Feature Sets**:
   - `EN1`: đúng 39 terms.
   - `EN2`: đúng 40 terms (hoặc 41 terms nếu có $P_c^2$).
   - `EN3`: đúng 54 terms (hoặc 55 terms nếu có $P_c^2$).
3. **File Artifacts xuất hiện đầy đủ trong `outputs/elastic_net/`**:
   - `executed_notebook.ipynb` (đầy đủ outputs ở tất cả các cells).
   - `models/selected_elastic_net_pipeline.joblib`.
   - `tables/cv_tuning_grid.csv`, `model_comparison.csv`, `selection_frequency.csv`.
   - `metrics.json` ghi nhận đầy đủ điểm số tối ưu và tỷ lệ thưa.
   - Tối thiểu 10 file hình ảnh PNG trong `figures/`.
4. **Kiểm tra Inference**:
   - Hàm `predict_student(raw_dict, pipeline)` nạp pipeline từ `.joblib` và dự đoán thành công cho một sinh viên mẫu.
   - Cảnh báo xuất hiện khi đưa vào danh mục lạ hoặc giá trị âm vô lý.
