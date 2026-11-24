# Contract: Model I/O và Artifacts — Elastic Net Regression

Contract này quy định giao diện lập trình, cấu trúc đối tượng pipeline và lược đồ định dạng các file artifacts đầu ra cho mô hình Elastic Net Regression.

---

## 1. Data Input Contract

| Kênh đầu vào | Định dạng | Yêu cầu kiểm tra & Ràng buộc |
| :--- | :--- | :--- |
| **Training Partition** | CSV / pandas DataFrame | Đọc từ `data/splits/train.csv` hoặc `data/train/dataset.csv`; đủ 35.000 dòng, chứa đúng 12 raw features + `Post_Semester_GPA`. |
| **Validation Partition** | CSV / pandas DataFrame | Đọc từ `data/splits/val.csv` hoặc `data/validation/dataset.csv`; đủ 7.500 dòng, chứa đúng 12 raw features + `Post_Semester_GPA`. |
| **Test Partition** | CSV / pandas DataFrame | Đọc từ `data/splits/test.csv` hoặc `data/test/dataset.csv`; chỉ kiểm tra Student_ID; **tuyệt đối không đọc target** trong giai đoạn này (`final test: not executed`). |
| **Single Row Inference** | Python dictionary / DataFrame 1 dòng | Đúng 12 key/cột raw features nhóm A–D. Báo lỗi `ValueError` nếu thiếu trường hoặc sai kiểu dữ liệu; cảnh báo nếu giá trị ngoài miền logic hoặc danh mục chưa thấy. |

---

## 2. Transformer Contract (`ElasticNetFeatures`)

Lớp `ElasticNetFeatures(model_type='EN3', scenario='full')` kế thừa từ `sklearn.base.BaseEstimator` và `TransformerMixin`:

```python
class ElasticNetFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, model_type='EN3', scenario='full', include_pc_squared=False):
        self.model_type = model_type
        self.scenario = scenario
        self.include_pc_squared = include_pc_squared

    def fit(self, X: pd.DataFrame, y=None):
        """Học các tham số chỉ từ tập train:
        - median của các biến số: H, P, S, Tool_Diversity, Paid_Subscription
        - mode của các biến phân loại
        - mean của H, P, S để phục vụ centering
        - OneHotEncoder(drop='first', handle_unknown='ignore')
        - Ghi nhận danh sách tên cột đầu ra get_feature_names_out()
        """
        ...
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Biến đổi ma trận đặc trưng:
        - Impute missing bằng median/mode đã học
        - Mã hóa One-Hot (unknown gán vector toàn 0 kèm warning)
        - Tính Hc = H - mean_H, Pc = P - mean_P, Sc = S - mean_S
        - Tạo đa thức phi tuyến (Hc^2; tùy chọn Pc^2)
        - Tạo các tương tác 2 chiều nếu model_type=='EN3'
        - Trả về ma trận 2D numpy array kiểu float64, hữu hạn
        """
        ...

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Trả về mảng 1D chứa tên của toàn bộ p đặc trưng đã sinh ra."""
        ...
```

---

## 3. Trained Pipeline Contract

Mô hình hoàn chỉnh được đóng gói bằng `sklearn.pipeline.Pipeline` với 3 bước tuần tự:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

pipeline = Pipeline([
    ('features', ElasticNetFeatures(model_type='EN3', scenario='full')),
    ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ('regressor', ElasticNet(alpha=alpha_opt, l1_ratio=l1_ratio_opt, random_state=42, max_iter=10000, tol=1e-4))
])
```

### Yêu cầu về Serialization:
- Lưu tại: `outputs/elastic_net/models/selected_elastic_net_pipeline.joblib`.
- Tương thích hoàn toàn với thư viện `joblib` tiêu chuẩn.
- Khi nạp lại bằng `joblib.load()`, đối tượng pipeline có thể nhận trực tiếp DataFrame raw 12 biến và gọi phương thức `.predict(X_raw)`.

---

## 4. Inference Function Contract (`predict_student`)

Giao diện dự đoán mức người dùng:

```python
def predict_student(raw_features: dict, model_pipeline: Pipeline) -> dict:
    """
    Args:
        raw_features: dictionary chứa đúng 12 trường dữ liệu thô.
        model_pipeline: Pipeline Elastic Net đã huấn luyện và fit hoàn chỉnh.
    Returns:
        dict: {
            'prediction': float,          # Điểm GPA dự đoán (không clip âm thầm)
            'model': str,                 # Tên mô hình (e.g. 'ElasticNet_EN3')
            'scenario': str,              # Kịch bản ('full' hoặc 'no_anxiety')
            'alpha': float,               # Siêu tham số alpha đã dùng
            'l1_ratio': float,            # Siêu tham số l1_ratio đã dùng
            'warnings': list[str]         # Danh sách cảnh báo (unknown category, out-of-domain)
        }
    """
```

---

## 5. Output Schemas & Artifacts Directory Structure

Toàn bộ kết quả thực thi được xuất vào thư mục `outputs/elastic_net/`:

```text
outputs/elastic_net/
├── executed_notebook.ipynb             # Notebook đã chạy với outputs đầy đủ
├── metrics.json                        # Lưu tổng hợp chỉ số tối ưu và kết quả đánh giá
├── models/
│   └── selected_elastic_net_pipeline.joblib  # Pipeline đã fit hoàn chỉnh
├── tables/
│   ├── cv_tuning_grid.csv              # Bảng kết quả lưới 5-fold CV (alpha, l1_ratio, metrics)
│   ├── model_comparison.csv            # Bảng so sánh EN1, EN2, EN3, Ablation vs Baseline OLS
│   ├── selection_frequency.csv         # Tần suất chọn biến qua 5 folds (0/5 đến 5/5)
│   ├── non_zero_coefficients.csv       # Danh sách hệ số khác 0 của mô hình tối ưu
│   └── hierarchy_check.csv             # Báo cáo kiểm tra tính phân cấp biến tương tác
└── figures/
    ├── 01_gpa_distribution.png         # Phân phối GPA tập train
    ├── 02_ai_hours_distribution.png    # Phân phối giờ dùng AI
    ├── 03_regularization_path.png      # Quỹ đạo hệ số theo alpha (Regularization Path)
    ├── 04_hyperparameter_heatmap.png   # Heatmap sai số CV theo (alpha, l1_ratio)
    ├── 05_cv_rmse_comparison.png       # Biểu đồ cột so sánh CV RMSE
    ├── 06_validation_rmse_r2.png       # Biểu đồ so sánh Validation RMSE và R2
    ├── 07_feature_selection_freq.png   # Tần suất chọn biến qua 5 folds
    ├── 08_coefficient_magnitudes.png   # Độ lớn các hệ số khác 0 được giữ lại
    ├── 09_residuals_vs_fitted.png      # Đồ thị phần dư kiểm tra giả định
    └── 10_what_if_response_curves.png  # Đồ thị phản hồi What-if theo giờ AI
```

### Schema chi tiết cho các file bảng chính:

#### `tables/cv_tuning_grid.csv`
- `model_type` (str): 'EN1', 'EN2', 'EN3', 'EN3_no_anxiety'
- `alpha` (float): Giá trị alpha thử nghiệm
- `l1_ratio` (float): Giá trị l1_ratio thử nghiệm
- `mean_cv_rmse` (float): RMSE trung bình qua 5 folds
- `std_cv_rmse` (float): Độ lệch chuẩn của RMSE
- `mean_cv_mae` (float): MAE trung bình
- `mean_cv_r2` (float): $R^2$ trung bình
- `mean_non_zero_terms` (float): Số hệ số khác 0 trung bình

#### `tables/model_comparison.csv`
- `model_name` (str): 'OLS_I0', 'OLS_I3', 'EN1', 'EN2', 'EN3', 'EN3_no_anxiety'
- `num_terms` (int): Tổng số predictor terms của thiết kế
- `non_zero_terms` (int): Số hệ số khác 0 thực tế sau phạt
- `sparsity_pct` (float): Tỷ lệ thưa (%)
- `val_rmse` (float): RMSE trên tập validation
- `val_mae` (float): MAE trên tập validation
- `val_mse` (float): MSE trên tập validation
- `val_r2` (float): $R^2$ trên tập validation
- `pct_out_of_bounds` (float): Tỷ lệ dự đoán ngoài $[0.0, 4.0]$ (%)

#### `metrics.json`
```json
{
  "run_id": "elastic_net_20261124",
  "created_at": "2026-11-24T...",
  "status": "validation_complete",
  "test_status": "not_executed",
  "selected_model": {
    "feature_set": "EN3",
    "alpha": 0.01,
    "l1_ratio": 0.7,
    "total_terms": 54,
    "non_zero_terms": 38,
    "sparsity_pct": 29.63
  },
  "validation_metrics": {
    "rmse": 0.1234,
    "mae": 0.0987,
    "mse": 0.0152,
    "r2": 0.8765,
    "below_zero_count": 0,
    "above_four_count": 12,
    "out_of_bounds_pct": 0.16
  },
  "ablation_metrics": {
    "no_anxiety_val_rmse": 0.1456,
    "no_anxiety_val_r2": 0.8321
  }
}
```
