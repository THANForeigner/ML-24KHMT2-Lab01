# Implementation Plan: Elastic Net Regression — Conditional GPA Prediction

**Branch**: `002-elastic-net-regression` | **Date**: 2026-11-24 | **Spec**: [spec.md](spec.md)

**Input**: Đặc tả kỹ thuật mô hình Elastic Net Regression, các quy ước kiến trúc của repository, và kinh nghiệm triển khai từ mô hình đối chứng `001-interaction-linear-regression`.

---

## 1. Summary

Xây dựng hệ thống huấn luyện và notebook tự học tiếng Việt cho mô hình **Elastic Net Regression** nhằm dự đoán `Post_Semester_GPA` (phân tích có điều kiện / What-if). Thiết kế kế thừa và tích hợp trực tiếp vào package `src/lab01_ml/`, sử dụng bộ chia dữ liệu 70/15/15 đã đóng băng trong `data/`, bảo vệ tuyệt đối tập holdout (`final test: not executed` khi chưa có biến môi trường `LAB_GROUP_SPLIT_APPROVED = 'yes'`). 

Toàn bộ các bước tiền xử lý: One-Hot Encoding drop-first, xử lý danh mục chưa thấy, tính mean để centering biến liên tục ($H_c = H - \overline{H}_{train}$), tạo đa thức phi tuyến/tương tác, và **chuẩn hóa thang đo (`StandardScaler`)** đều được đóng gói khép kín trong `Pipeline` của `scikit-learn` và fit riêng biệt trong từng fold Cross-Validation trên tập `train`. Quá trình tuning thực hiện tìm kiếm lưới trên cặp siêu tham số $(\alpha, l_1\_ratio)$ qua 5-fold CV, áp dụng tiêu chuẩn hòa (tie-break) ưu tiên độ thưa (sparsity) cao hơn. Mô hình được đánh giá trên tập `validation` độc lập, so sánh với baseline OLS, trích xuất tần suất chọn biến, kiểm tra tính phân cấp (hierarchy), phát hiện dự đoán vượt trần GPA 4.0, và xuất pipeline hoàn chỉnh dạng `.joblib` nhận đúng 12 raw features.

---

## 2. Technical Context

- **Language/Runtime**: Python $\ge 3.11$; Notebook nbformat 4.
- **Core Dependencies**: `scikit-learn` (Pipeline, OneHotEncoder, StandardScaler, ElasticNet, KFold), `numpy`, `pandas`, `matplotlib`, `seaborn`, `joblib`, `nbformat`, `nbclient`, `pytest`.
- **Target Platform**: Windows, Linux/WSL, Kaggle, Colab (môi trường CPU chuẩn; không yêu cầu GPU hay kết nối mạng khi runtime).
- **Storage & Artifacts**:
  - Không gian lưu trữ: `outputs/elastic_net/` (gồm `executed_notebook.ipynb`, `models/`, `tables/`, `figures/`, `metrics.json`).
  - Dữ liệu nguồn: Đọc trực tiếp từ `data/` (`assignments.csv`, `metadata.json`, `data/train/dataset.csv` hoặc `data/splits/train.csv`). Tuyệt đối không tự tạo lại hay ghi đè split.
- **Testing**: `pytest`, `unittest`, kiểm tra rò rỉ dữ liệu (leakage guards), kiểm tra số terms (39/40/54), kiểm tra fold-only scaling, kiểm tra serialization, và clean-kernel notebook execution.
- **Constraints**:
  - 12 raw features đầu vào, 1 target duy nhất (`Post_Semester_GPA`). Loại trừ hoàn toàn `Student_ID`, `GPA_Change`, `Skill_Retention_Score`, `Burnout_Risk_Level`.
  - Không truy cập target của tập `test` trong quá trình tuning; chỉ mở khóa khi có cờ `LAB_GROUP_SPLIT_APPROVED = 'yes'`.
  - Không giữ đồng thời biến thô $H$ và biến trừ tâm $H_c$ khi mô hình có hệ số chặn (intercept).
  - Không tự ý dùng hàm cắt `clip()` âm thầm cho dự đoán ngoài $[0.0, 4.0]$.

---

## 3. Constitution & Architecture Gates

| Kiểm tra cổng kiến trúc | Trạng thái & Cơ chế thực thi |
| :--- | :--- |
| **Academic Integrity & No Causation** | Dữ liệu synthetic; tài liệu và notebook luôn nhấn mạnh mối liên hệ tương quan có điều kiện (conditional association), không suy diễn nhân quả. |
| **Split Reuse & No Overwrite** | Nạp từ dữ liệu đã đóng băng, đối chiếu SHA-256 với `data/metadata.json` và `data/assignments.csv`. Không resplit, không chuyển dòng giữa các tập. |
| **Strict Holdout Protection** | Mặc định không đọc target của tập test. Khi chạy pipeline chỉ tính CV và validation; `final test: not executed`. Chỉ đánh giá test một lần duy nhất sau khi model lock và có `$env:LAB_GROUP_SPLIT_APPROVED = 'yes'`. |
| **Pipeline Encapsulation & No Leakage** | `Pipeline([('features', ElasticNetFeatures()), ('scaler', StandardScaler()), ('regressor', ElasticNet())])`. Tất cả thống kê (median, mode, mean, std) fit trong từng fold của CV. |
| **Standardization for Penalization** | Do hàm phạt Elastic Net đánh vào tổng độ lớn hệ số $\alpha(l_1 \|\beta\|_1 + \frac{1-l_1}{2}\|\beta\|_2^2)$, toàn bộ các đặc trưng sau mở rộng bắt buộc phải chuẩn hóa bằng `StandardScaler` trước bộ ước lượng. |
| **Reproducibility** | Cố định `random_state=42` ở split, CV folds và `ElasticNet`. Kết quả lặp lại chính xác 100%. |

---

## 4. Project Structure & Codebase Integration

Hệ thống mở rộng trực tiếp cấu trúc của `src/lab01_ml/`, tái sử dụng tối đa mã nguồn chung và bổ sung các module chuyên biệt cho Elastic Net:

```text
ML-24KHMT2-Lab01/
├── data/
│   ├── assignments.csv                       # Phân vùng Student_ID -> split
│   ├── metadata.json                          # Mã băm SHA-256 các tập dữ liệu
│   ├── train/dataset.csv                      # Tập train (35.000 dòng)
│   ├── validation/dataset.csv                 # Tập validation (7.500 dòng)
│   └── test/dataset.csv                       # Tập test (7.500 dòng - holdout)
├── notebooks/
│   ├── interaction_linear_regression.ipynb   # Model 001
│   └── elastic_net_regression.ipynb          # Model 002 (Deliverable chính)
├── outputs/
│   └── elastic_net/                           # Toàn bộ outputs của Model 002
│       ├── executed_notebook.ipynb            # Notebook đã chạy với outputs
│       ├── metrics.json                       # Chỉ số CV, Validation, Sparsity, Test status
│       ├── model_lock.json                    # Khóa cấu hình mô hình trước test
│       ├── models/
│       │   └── selected_elastic_net_pipeline.joblib  # Pipeline đã fit hoàn chỉnh
│       ├── tables/
│       │   ├── cv_tuning_grid.csv             # Kết quả lưới siêu tham số 5-fold CV
│       │   ├── model_comparison.csv           # So sánh EN1, EN2, EN3, Ablation vs OLS
│       │   ├── selection_frequency.csv        # Tần suất chọn biến qua 5 folds
│       │   ├── non_zero_coefficients.csv      # Hệ số khác 0 của mô hình tối ưu
│       │   └── hierarchy_check.csv            # Báo cáo kiểm tra tính phân cấp
│       └── figures/                           # 10+ biểu đồ PNG (300 DPI)
├── scripts/
│   ├── build_interaction_notebook.py          # Script build của nhánh 001
│   ├── execute_interaction_notebook.py        # Script execute của nhánh 001
│   ├── build_elastic_net_notebook.py          # Script tạo notebook Elastic Net 34 phần
│   └── execute_elastic_net_notebook.py        # Script chạy notebook trong clean kernel
├── specs/
│   ├── 001-interaction-linear-regression/     # Specs nhánh 001
│   └── 002-elastic-net-regression/            # Specs nhánh 002
│       ├── spec.md                            # Đặc tả kỹ thuật
│       ├── plan.md                            # Kế hoạch triển khai kiến trúc
│       ├── data-model.md                      # Lược đồ dữ liệu và thực thể
│       ├── research.md                        # Khảo sát lý thuyết và quyết định kỹ thuật
│       ├── quickstart.md                      # Hướng dẫn chạy và nghiệm thu
│       ├── contracts/model-io.md              # Hợp đồng I/O và Artifacts
│       └── checklists/requirements.md         # Checklist chất lượng đặc tả
├── src/
│   └── lab01_ml/
│       ├── data/
│       │   ├── loader.py                      # Nạp dữ liệu linh hoạt (splits/ hoặc train/)
│       │   └── preprocessing.py               # Các helper làm sạch cơ bản
│       ├── features/
│       │   ├── engineering.py                 # Transformer tương tác & đa thức (tái sử dụng)
│       │   └── elastic_net_features.py        # Transformer EN1/EN2/EN3 chuyên biệt
│       ├── models/
│       │   ├── factory.py                     # Khởi tạo Pipeline (Features -> Scaler -> Model)
│       │   ├── training.py                    # Vòng lặp CV & development OLS
│       │   ├── elastic_net_training.py        # 5-fold CV tuning (alpha, l1_ratio) & ablation
│       │   ├── elastic_net_evaluation.py      # Sparsity, frequency, hierarchy, ceiling check
│       │   └── inference.py                   # Hàm predict_student cho 12 raw features
│       └── visualization/
│           └── plots.py                       # Helper tạo các biểu đồ chuẩn mực
└── tests/
    ├── test_features.py                       # Unit tests features sẵn có
    ├── test_elastic_net_features.py           # Unit tests kiểm tra EN1(39), EN2(40), EN3(54)
    ├── test_elastic_net_training.py           # Unit tests CV tuning, tie-break, ablation
    ├── test_elastic_net_inference.py          # Unit tests inference, warnings, reload joblib
    └── test_leakage_and_splits.py             # Unit tests tính toàn vẹn và no-leakage
```

---

## 5. Architectural Components & Workflow

### 5.1. Data Loading & Leakage Guard (`data/loader.py`)
- Mở rộng hàm `load_frozen(root)`:
  - Hỗ trợ tìm kiếm thông minh: kiểm tra cả đường dẫn `data/splits/{train.csv, val.csv, test.csv}` và cấu trúc `data/{train/dataset.csv, validation/dataset.csv, test/dataset.csv}`.
  - Xác thực mã băm SHA-256 đối chiếu với `data/metadata.json` và `data/assignments.csv`.
  - Đảm bảo tính rời rạc 100% của `Student_ID`: $\text{IDs}_{train} \cap \text{IDs}_{val} = \emptyset$, $\text{IDs}_{train} \cap \text{IDs}_{test} = \emptyset$, $\text{IDs}_{val} \cap \text{IDs}_{test} = \emptyset$.
  - Khi nạp tập `test`, chỉ đọc cột `Student_ID` để kiểm tra phân vùng; không đọc cột `Post_Semester_GPA` trong suốt quá trình huấn luyện và chọn mô hình.

### 5.2. Transformer & Feature Sets (`features/elastic_net_features.py`)
Xây dựng lớp `ElasticNetFeatures(model_type='EN3', scenario='full', include_pc_squared=False)` tuân thủ chuẩn `BaseEstimator` và `TransformerMixin`:
1. **Fit**:
   - Tính toán median của các biến số (`Weekly_GenAI_Hours`, `Pre_Semester_GPA`, `Traditional_Study_Hours`, `Tool_Diversity`, `Paid_Subscription`).
   - Tính mode của các biến phân loại.
   - Tính trung bình của $H$, $P$, $S$ trên tập train để làm mốc trừ tâm.
   - Fit `OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)`.
   - Sinh danh sách tên đặc trưng `get_feature_names_out()`.
2. **Transform**:
   - Impute missing bằng median/mode đã học.
   - Mã hóa One-Hot (unknown categories chuyển thành vector toàn 0 kèm UserWarning).
   - Trừ tâm: $H_c = H - \overline{H}_{train}$, $P_c = P - \overline{P}_{train}$, $S_c = S - \overline{S}_{train}$.
   - Sinh các thành phần đặc trưng theo `model_type`:
     - **EN1 (39 terms)**: $[H_c, P_c, S_c, \text{Tool\_Diversity}, \text{Paid\_Subscription}, \text{34 dummy columns}]$.
     - **EN2 (40–41 terms)**: Bổ sung $H_c^2$ (và tùy chọn $P_c^2$). **Không giữ biến $H$ thô** khi có intercept.
     - **EN3 (54–55 terms)**: Bổ sung 14 terms tương tác 2 chiều ($4 \times H_c \cdot U_i$, $2 \times H_c \cdot K_j$, $8 \times U_i \cdot K_j$).
     - **EN3_no_anxiety (45–46 terms)**: Loại bỏ biến `Anxiety_Level_During_Exams`.
   - Kiểm tra giá trị hữu hạn (`np.isfinite(result).all()`).

### 5.3. Pipeline & Scaling Architecture (`models/factory.py`)
Mỗi mô hình Elastic Net được tích hợp thành một `Pipeline` duy nhất gồm 3 bước:
```python
Pipeline([
    ('features', ElasticNetFeatures(model_type=model_type, scenario=scenario)),
    ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ('regressor', ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=10000, tol=1e-4))
])
```
*Lưu ý*: Mọi bước tiền xử lý và scaling đều diễn ra bên trong fold của Cross-Validation thông qua `clone(pipeline).fit(X_tr, y_tr)`.

### 5.4. Hyperparameter Tuning & Cross-Validation (`models/elastic_net_training.py`)
Hàm `tune_elastic_net(X_train, y_train, model_type='EN3', scenario='full')`:
- Thiết lập lưới tìm kiếm:
  - $\alpha \in [10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0]$
  - $l_1\_ratio \in [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]$
- Chia 5 folds ngẫu nhiên cố định: `KFold(n_splits=5, shuffle=True, random_state=42)`.
- Chạy đánh giá cho từng cặp $(\alpha, l_1\_ratio)$:
  - Ghi nhận `mae`, `mse`, `rmse`, `r2` trên từng fold.
  - Tính mean và sample standard deviation (ddof=1) qua 5 folds.
  - Đếm số lượng hệ số khác 0 và tính tần suất chọn biến.
- Áp dụng chính sách chọn mô hình:
  - Tìm $\text{min\_rmse} = \min(\text{mean\_cv\_rmse})$.
  - Tập các mô hình thỏa mãn $\text{mean\_cv\_rmse} \le 1.001 \times \text{min\_rmse}$ được đưa vào diện cân nhắc.
  - Trong tập này, chọn mô hình có số hệ số khác 0 trung bình ít nhất (độ thưa lớn nhất).
  - Nếu vẫn hòa, chọn mô hình có $l_1\_ratio$ cao hơn (ưu tiên phạt thưa).
- Fit mô hình tối ưu trên toàn bộ 35.000 dòng train với cặp $(\alpha^*, l_1\_ratio^*)$.

### 5.5. Group Ablation & Đánh giá Validation (`models/elastic_net_training.py`)
- Chạy quy trình tuning tương tự cho biến thể `EN3_no_anxiety` (loại bỏ `Anxiety_Level_During_Exams`).
- Đánh giá tất cả các mô hình đã fit (`EN1`, `EN2`, `EN3`, `EN3_no_anxiety`) trên tập `validation` (7.500 dòng).
- So sánh định lượng với baseline OLS (`I0` và `I3` từ mô hình tương tác) trên cùng tập validation:
  - RMSE, MAE, MSE, $R^2$.
  - Số lượng hệ số khác 0 và tỷ lệ thưa.

### 5.6. Elastic Net Specific Diagnostics (`models/elastic_net_evaluation.py`)
1. **Sparsity & Coefficients**:
   - Trích xuất hệ số từ pipeline: `coef = pipeline.named_steps['regressor'].coef_`.
   - Đếm số hệ số khác 0 ($|\beta_j| > 10^{-6}$), lập bảng các hệ số quan trọng nhất.
2. **Selection Frequency**:
   - Thống kê tỷ lệ fold mà đặc trưng $j$ có hệ số khác 0 ($c_j / 5$).
   - Vẽ biểu đồ cột phân loại các đặc trưng: Cốt lõi ($5/5$), Không ổn định ($2/5 - 4/5$), Bị triệt tiêu ($0/5 - 1/5$).
3. **Hierarchy Check**:
   - Kiểm tra các cặp tương tác $H_c \cdot U_i$, $H_c \cdot K_j$, $U_i \cdot K_j$.
   - Phát hiện các trường hợp vi phạm phân cấp: biến tương tác có hệ số $\neq 0$ nhưng biến thành phần bị ép về 0.
4. **GPA Ceiling / Floor Check**:
   - Quét dự đoán trên tập validation: đếm số mẫu $\hat{y} > 4.0$ hoặc $\hat{y} < 0.0$.
   - Tính tỷ lệ phần trăm out-of-bounds; tuyệt đối không tự ý clip.

### 5.7. Model Lock & Final Test Guard
- Lưu cấu hình mô hình tối ưu vào `outputs/elastic_net/model_lock.json`.
- Xuất pipeline hoàn chỉnh ra `outputs/elastic_net/models/selected_elastic_net_pipeline.joblib`.
- Ghi nhận `outputs/elastic_net/metrics.json`:
  ```json
  {
    "status": "validation_complete",
    "final_test": {"status": "not executed", "reason": "group split approval missing"}
  }
  ```
- **Cơ chế mở khóa**: Chỉ khi biến môi trường `LAB_GROUP_SPLIT_APPROVED == 'yes'`, script mới tiến hành nạp tập test, dự đoán bằng model đã khóa, và cập nhật `final_test` trong `metrics.json`.

---

## 6. Notebook Outline (34 phần chi tiết bằng tiếng Việt)

Notebook `notebooks/elastic_net_regression.ipynb` được sinh tự động bởi `scripts/build_elastic_net_notebook.py` gồm 34 phần hoàn chỉnh:

1. **01. Mục tiêu & Động cơ**: Giới thiệu bài toán Conditional GPA Prediction, hiện tượng đa cộng tuyến khi mở rộng đặc trưng, và lý do chọn Elastic Net.
2. **02. Thiết lập Môi trường**: Cấu hình đường dẫn tự động, nạp các thư viện, thiết lập style đồ thị Seaborn.
3. **03. Hợp đồng Dữ liệu Đầu vào**: Liệt kê 12 raw features thuộc nhóm A–D và 1 target `Post_Semester_GPA`.
4. **04. Quy tắc Ngăn chặn Rò rỉ (Leakage Guard)**: Khẳng định việc loại bỏ 4 cột cấm (`Student_ID`, `GPA_Change`, `Skill_Retention_Score`, `Burnout_Risk_Level`).
5. **05. Phân tích Thời điểm Đo lường (Timing & Retrospective)**: Thảo luận về rủi ro rò rỉ thời gian của biến `Anxiety_Level_During_Exams`.
6. **06. Kiểm tra Split Đóng băng & Mã băm**: Nạp 3 tập dữ liệu qua `load_frozen()`, đối chiếu hash với `data/metadata.json`.
7. **07. Kiểm tra Sơ đồ & Dữ liệu Thiếu**: Khảo sát missing values và dtype trên tập train.
8. **08. Phân phối Target GPA (Train)**: Histogram và KDE phân phối `Post_Semester_GPA` (Hình 01).
9. **09. Phân phối Giờ dùng AI (Train)**: Histogram số giờ dùng AI `Weekly_GenAI_Hours` (Hình 02).
10. **10. Tương quan Thô & GPA Change**: Biểu đồ phân tán giữa AI Hours và thay đổi GPA minh họa (Hình 03).
11. **11. Phân tích Nhóm Use Case × Prompt Skill**: Heatmap GPA trung bình theo nhóm (Hình 04).
12. **12. Lý thuyết Toán học Elastic Net**: Công thức hàm mất mát $L_1 + L_2$, cơ chế Coordinate Descent và Grouping Effect.
13. **13. Tầm quan trọng của StandardScaler**: Giải thích tại sao bắt buộc chuẩn hóa thang đo khi phạt hồi quy.
14. **14. Centering & Quy tắc Trừ tâm**: Công thức $H_c = H - \overline{H}_{train}$ và quy tắc không giữ đồng thời $H$ thô khi có intercept.
15. **15. One-Hot Encoding drop-first & Reference Category**: Bảng danh mục tham chiếu và cơ chế xử lý danh mục lạ.
16. **16. Hệ phân cấp Feature Sets (EN1, EN2, EN3)**: So sánh số terms: 39 terms (EN1), 40–41 terms (EN2), 54–55 terms (EN3).
17. **17. Minh họa Tính toán Thủ công một hàng**: So sánh giá trị biến đổi tay và output của `ElasticNetFeatures`.
18. **18. Thiết lập 5-Fold Cross-Validation**: Phân chia 5 folds cố định với `random_state=42`, cam kết fit transformer/scaler riêng từng fold.
19. **19. Tinh chỉnh Siêu tham số (Grid Search)**: Chạy lưới $(\alpha, l_1\_ratio)$ cho `EN1`, `EN2`, `EN3`.
20. **20. Bề mặt Sai số Siêu tham số**: Heatmap Validation RMSE theo dải $(\alpha, l_1\_ratio)$ (Hình 05).
21. **21. Quỹ đạo Hệ số hồi quy (Regularization Path)**: Đồ thị thể hiện các hệ số bị co về 0 khi $\alpha$ tăng (Hình 06).
22. **22. So sánh Sai số CV (CV RMSE)**: Biểu đồ cột so sánh sai số CV $\pm \text{sd}$ giữa các cấu hình (Hình 07).
23. **23. Đánh giá trên tập Validation**: So sánh định lượng `EN1`, `EN2`, `EN3` và OLS baseline trên 7.500 mẫu validation (Hình 08).
24. **24. Group Ablation (Loại bỏ Anxiety)**: So sánh Full vs No-Anxiety trên tập validation và đánh đổi dự báo trước kỳ thi.
25. **25. Trích xuất Độ thưa & Hệ số Tối ưu**: Thống kê số lượng hệ số khác 0 ($k$) và biểu đồ độ lớn các hệ số được giữ lại (Hình 09).
26. **26. Tần suất Chọn Biến qua 5 Folds**: Biểu đồ cột tỷ lệ xuất hiện của các biến qua 5 folds ($0/5 \dots 5/5$) (Hình 10).
27. **27. Kiểm tra Tính Phân cấp (Hierarchy Check)**: Phân tích các cặp tương tác có vi phạm nguyên lý phân cấp cấu trúc hay không.
28. **28. Kiểm soát Trần Điểm GPA (Ceiling / Floor)**: Đếm và báo cáo tỷ lệ dự đoán ngoài $[0.0, 4.0]$; khẳng định không dùng `clip()`.
29. **29. Chẩn đoán Phần dư (Residual Diagnostics)**: Đồ thị Residuals vs Fitted trên tập validation kiểm tra phương sai sai số (Hình 11).
30. **30. Khóa Mô hình & Final Test Guard**: Khóa `model_lock.json`, lưu model joblib, kiểm tra cờ `LAB_GROUP_SPLIT_APPROVED`.
31. **31. Đánh giá Final Holdout (Điều kiện)**: Chỉ chạy khi được phê duyệt; nếu chưa có thì in `final test: not executed`.
32. **32. Dự đoán Đơn lẻ (Raw Profile Inference)**: Thử nghiệm hàm `predict_student()` với 1 sinh viên mẫu từ raw features.
33. **33. Phân tích Kịch bản What-if & Giao diện Tương tác**: Đồ thị phản hồi GPA dự đoán khi quét giờ AI từ 0 đến 40 giờ (Hình 12) kèm widget fallback.
34. **34. Thảo luận Giới hạn & Khuyến nghị**: Nêu rõ tính chất synthetic, kịch bản retrospective, và giới hạn term-level selection của Elastic Net.

---

## 7. Test Strategy Matrix

| Test Suite | File kiểm thử | Yêu cầu kiểm chứng |
| :--- | :--- | :--- |
| **Feature Dimensions & Hierarchy** | `tests/test_elastic_net_features.py` | - `EN1` tạo đúng 39 terms khi đủ levels.<br>- `EN2` tạo đúng 40 terms (hoặc 41 terms nếu có $P_c^2$), không trùng raw $H$.<br>- `EN3` tạo đúng 54 terms (hoặc 55 terms) với đủ 14 interaction terms.<br>- `EN3_no_anxiety` tạo đúng 45 terms (hoặc 46 terms). |
| **Leakage & Preprocessing Isolation** | `tests/test_leakage_and_splits.py` | - Kiểm tra mã băm SHA-256 các tập dữ liệu.<br>- Xác nhận `Student_ID` hoàn toàn rời rạc giữa train, val, test.<br>- Xác minh `StandardScaler` và `InteractionFeatures` chỉ học trung bình/phương sai từ train fold.<br>- Kiểm tra thay đổi dữ liệu test không làm thay đổi trọng số đã fit. |
| **Tuning & Training Logic** | `tests/test_elastic_net_training.py` | - 5-fold CV tái lập 100% với `random_state=42`.<br>- Tiêu chí tie-break $\le 0.1\%$ ưu tiên mô hình có ít hệ số khác 0 hơn.<br>- Hàm ablation hoạt động chính xác khi loại bỏ Anxiety. |
| **Sparsity & Diagnostics** | `tests/test_elastic_net_training.py` | - Trích xuất đúng số hệ số khác 0 và tính toán sparsity rate.<br>- Tần suất chọn biến nằm trong đoạn $[0.0, 1.0]$.<br>- Báo cáo hierarchy check phát hiện đúng các cặp vi phạm. |
| **Inference & Serialization** | `tests/test_elastic_net_inference.py` | - Hàm `predict_student` nhận đúng dict 12 raw features và trả kết quả hợp lệ.<br>- Báo lỗi `ValueError` khi thiếu trường hoặc sai kiểu.<br>- Đưa ra `UserWarning` khi gặp unknown category hoặc giá trị ngoài miền.<br>- Nạp lại file `.joblib` từ đĩa và dự đoán cho kết quả trùng khớp sai số $< 10^{-9}$. |
| **Notebook Clean Execution Smoke** | `tests/test_notebook_smoke.py` | - Kiểm tra notebook `notebooks/elastic_net_regression.ipynb` hợp lệ theo chuẩn `nbformat 4`.<br>- Chạy thử nghiệm smoke trong kernel sạch. |

---

## 8. Implementation Steps & Milestones

1. **Bước 1 — Core Modules (`src/lab01_ml/`)**:
   - Cập nhật `src/lab01_ml/data/loader.py` để hỗ trợ linh hoạt các đường dẫn split hiện có (`data/splits/` hoặc `data/{train,val,test}/`).
   - Cài đặt `ElasticNetFeatures` trong `src/lab01_ml/features/elastic_net_features.py` hoặc mở rộng `features/engineering.py`.
   - Cài đặt hàm dựng pipeline trong `src/lab01_ml/models/factory.py`.
   - Cài đặt hàm tuning 5-fold CV và ablation trong `src/lab01_ml/models/elastic_net_training.py`.
   - Cài đặt hàm trích xuất chẩn đoán (sparsity, selection frequency, hierarchy check, ceiling check) trong `src/lab01_ml/models/elastic_net_evaluation.py`.
   - Cập nhật hàm `predict_student` trong `src/lab01_ml/models/inference.py` hỗ trợ mô hình Elastic Net.
2. **Bước 2 — Unit Tests (`tests/`)**:
   - Viết các test suite tương ứng trong `tests/` để kiểm tra toàn bộ tính năng và bảo vệ chống rò rỉ dữ liệu.
   - Chạy kiểm thử tự động với `pytest`.
3. **Bước 3 — Scripts Tạo & Chạy Notebook**:
   - Viết `scripts/build_elastic_net_notebook.py` tự động dựng cấu trúc notebook 34 phần với đầy đủ văn bản tiếng Việt, công thức LaTeX và code cells.
   - Viết `scripts/execute_elastic_net_notebook.py` sử dụng `nbclient` để chạy notebook trong kernel sạch, xuất artifacts ra `outputs/elastic_net/`.
4. **Bước 4 — Thực thi & Xuất Artifacts**:
   - Chạy build và execute notebook.
   - Kiểm tra các biểu đồ PNG, bảng CSV, file `metrics.json` và file model `selected_elastic_net_pipeline.joblib`.
5. **Bước 5 — Kiểm tra Nghiệm thu & Tài liệu hóa**:
   - Đối chiếu với checklist nghiệm thu trong `quickstart.md` và `spec.md`.
