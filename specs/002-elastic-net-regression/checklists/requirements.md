# Specification Quality Checklist: Elastic Net Regression — Conditional GPA Prediction

**Purpose**: Validate specification completeness, mathematical rigor, and leakage-prevention quality before proceeding to implementation planning  
**Created**: 2026-11-24  
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] Không rò rỉ chi tiết cài đặt khung nội bộ; các deliverables (notebook, package module, unit tests, pipeline serialized) đều là các giao diện người dùng và ứng dụng rõ ràng.
- [x] Tập trung vào giá trị người học và mục tiêu phân tích điều kiện (Conditional GPA Prediction / What-if Analysis).
- [x] Diễn đạt chuẩn xác các thuật ngữ thống kê và học máy: Regularization, $L_1$ Penalty (Lasso), $L_2$ Penalty (Ridge), Coordinate Descent, Shrinkage, Sparsity, Grouping Effect, Standardization.
- [x] Mọi mục bắt buộc của đặc tả (Clarifications, User Scenarios, Requirements, Success Criteria, Assumptions, Edge Cases) đều hoàn thiện.

## Requirement Completeness

- [x] Không còn thẻ `[NEEDS CLARIFICATION]` nào chưa được giải quyết; các vấn đề mở đã được ghi rõ trong mục Open Issues.
- [x] Yêu cầu đầu vào và đầu ra hoàn toàn tường minh: đúng 12 raw features thuộc nhóm A–D, target duy nhất là `Post_Semester_GPA`, 4 cột bị loại bỏ hoàn toàn.
- [x] Tiêu chí thành công có thể đo lường định lượng (SC-001 đến SC-010).
- [x] Các kịch bản nghiệm thu (Acceptance Scenarios) theo cấu trúc Given-When-Then bao phủ toàn bộ 10 User Stories.
- [x] Nhận diện và quy định cách xử lý đầy đủ các trường hợp biên (Edge Cases): tất cả hệ số về 0, cực biên Lasso/Ridge, cảnh báo hội tụ, thiếu cấp độ phân loại, dữ liệu ngoài miền.
- [x] Phạm vi nghiên cứu (Scope) và giới hạn (Out of Scope) được phân định rõ ràng.

## Feature Readiness

- [x] Mọi yêu cầu chức năng (FR-001 đến FR-024) đều có tiêu chí nghiệm thu rõ ràng.
- [x] Đã thiết lập cơ chế khóa tập test (`final test: not executed`) để bảo vệ tính độc lập của tập holdout tương tự mô hình `001-interaction-linear-regression`.
- [x] Toàn bộ quy trình tiền xử lý, biến đổi đặc trưng và `StandardScaler` được yêu cầu đóng gói khép kín trong `Pipeline`.

---

## Checklist chuyên đề: Lab 01, Elastic Net Regularization và ML Leakage Prevention

**Mục đích**: Thẩm định chi tiết các khía cạnh chuyên sâu về rò rỉ dữ liệu, chuẩn hóa thang đo, cấu trúc đặc trưng EN1–EN3, tối ưu hóa siêu tham số và trích xuất chỉ số thưa trước khi bắt đầu lập kế hoạch thực hiện.

### 1. Hợp đồng Dữ liệu & Ngăn ngừa Leakage

- [x] **CHK001**: Target `Post_Semester_GPA`, đúng 12 raw features thuộc nhóm A–D đã được định nghĩa đầy đủ, nhất quán giữa spec và repo? [Completeness; Spec: FR-002](../spec.md#functional-requirements)
- [x] **CHK002**: Danh sách loại trừ tuyệt đối (`Student_ID`, `GPA_Change`, `Skill_Retention_Score`, `Burnout_Risk_Level`) được thực thi nghiêm ngặt, ngăn chặn triệt để target leakage? [Integrity; Spec: FR-003](../spec.md#functional-requirements)
- [x] **CHK003**: Dữ liệu nạp từ bộ chia sẵn 70/15/15 đã đóng băng, đối chiếu mã băm với `data/assignments.csv` và `data/metadata.json`, tuyệt đối không tự ý chia lại? [Integrity; Spec: FR-004](../spec.md#functional-requirements)
- [x] **CHK004**: Trạng thái tập `test` được bảo vệ nghiêm ngặt (`final test: not executed`), không đọc target trong suốt quá trình huấn luyện và chọn siêu tham số? [Holdout Protection; Spec: FR-005](../spec.md#functional-requirements)
- [x] **CHK005**: Mọi tham số tiền xử lý (imputation median/mode, One-Hot categories, centering mean, và `StandardScaler` mean/variance) chỉ được học từ training fold tương ứng trong CV? [No Leakage; Spec: FR-006, FR-010](../spec.md#functional-requirements)

### 2. Không gian Đặc trưng & Hệ phân cấp (EN1, EN2, EN3)

- [x] **CHK006**: Tập đặc trưng `EN1` gồm đúng 39 terms sau One-Hot Encoding drop-first (5 numeric/binary + 34 dummy columns) khi dữ liệu có đủ levels? [Completeness; Spec: FR-007](../spec.md#functional-requirements)
- [x] **CHK007**: Tập đặc trưng `EN2` gồm 40–41 terms, bổ sung $H_c^2$ (và tùy chọn $P_c^2$), loại bỏ biến thô $H$ khi có intercept để tránh hoàn toàn đa cộng tuyến hoàn hảo? [Collinearity Prevention; Spec: FR-008](../spec.md#functional-requirements)
- [x] **CHK008**: Tập đặc trưng `EN3` gồm 54–55 terms, bổ sung đủ 14 interaction terms thuộc 3 nhóm ($H_c \times \text{Use Case}$, $H_c \times \text{Prompt Skill}$, $\text{Use Case} \times \text{Prompt Skill}$) và giữ nguyên vẹn toàn bộ main effects? [Hierarchy Integrity; Spec: FR-009](../spec.md#functional-requirements)
- [x] **CHK009**: Quy tắc mã hóa danh mục lạ (unknown categories) được xử lý an toàn: gán vector 0 (tương đương reference) kèm theo cảnh báo rõ ràng? [Robustness; Spec: User Story 3, Edge Cases](../spec.md#user-story-3--xây-dựng-pipeline--feature-sets-en1en3-priority-p1)

### 3. Chuẩn hóa Thang đo (StandardScaler) cho Regularization

- [x] **CHK010**: Có yêu cầu bắt buộc chuẩn hóa thang đo (`StandardScaler`) cho toàn bộ các đặc trưng trước khi áp dụng hàm phạt Elastic Net không? [Mathematical Rigor; Spec: FR-010](../spec.md#functional-requirements)
- [x] **CHK011**: `StandardScaler` được tích hợp bên trong `Pipeline` của `scikit-learn` để đảm bảo transform tự động và không rò rỉ phương sai giữa các fold? [Pipeline Encapsulation; Spec: FR-006, FR-010](../spec.md#functional-requirements)

### 4. Tinh chỉnh Siêu tham số & Đánh giá

- [x] **CHK012**: Tinh chỉnh đồng thời cặp siêu tham số $(\alpha, l_1\_ratio)$ bằng 5-fold Cross-Validation trên tập `train` cho cả `EN1`, `EN2`, `EN3`? [Completeness; Spec: FR-011](../spec.md#functional-requirements)
- [x] **CHK013**: Không gian tìm kiếm lưới siêu tham số đủ rộng ($\alpha \in [10^{-4}, 10^1]$, $l_1\_ratio \in [0.1, 1.0]$)? [Optimization; Spec: FR-011](../spec.md#functional-requirements)
- [x] **CHK014**: Quy tắc chọn mô hình tối ưu theo mean CV RMSE và chính sách hòa (tie-break $\le 0.1\%$) ưu tiên độ thưa / mô hình đơn giản được quy định rõ ràng? [Objectivity; Spec: FR-012](../spec.md#functional-requirements)
- [x] **CHK015**: Phiên bản Group Ablation loại bỏ `Anxiety_Level_During_Exams` được huấn luyện và đánh giá trên cùng split/CV để kiểm tra rủi ro temporal leakage? [Sensitivity Analysis; Spec: FR-013](../spec.md#functional-requirements)
- [x] **CHK016**: Đánh giá trên tập `validation` gồm đủ 4 chỉ số RMSE, MAE, MSE, $R^2$ và so sánh định lượng trực tiếp với baseline OLS? [Comparative Evaluation; Spec: FR-014](../spec.md#functional-requirements)

### 5. Chỉ số Đặc thù Elastic Net & Giới hạn Nghiên cứu

- [x] **CHK017**: Trích xuất số hệ số khác 0 ($k$) và tỷ lệ thưa ($\%$) của mô hình tối ưu? [Sparsity Diagnostics; Spec: FR-015](../spec.md#functional-requirements)
- [x] **CHK018**: Tính toán và trực quan hóa tần suất chọn biến (selection frequency) qua 5 fold của CV để đánh giá độ ổn định của các đặc trưng? [Stability Analysis; Spec: FR-015](../spec.md#functional-requirements)
- [x] **CHK019**: Kiểm tra tính phân cấp (hierarchy check): ghi nhận hiện tượng biến tương tác được giữ lại khi biến thành phần bị triệt tiêu về 0? [Methodological Rigor; Spec: FR-015](../spec.md#functional-requirements)
- [x] **CHK020**: Theo dõi và báo cáo tỷ lệ dự đoán ngoài miền $[0.0, 4.0]$ (ceilling/floor check), tuyệt đối không cắt/clip âm thầm trước khi tính sai số? [Calibration Transparency; Spec: FR-016](../spec.md#functional-requirements)
- [x] **CHK021**: Nêu rõ 3 giới hạn cốt lõi: dữ liệu synthetic (không suy luận nhân quả), kịch bản hồi cứu (retrospective), và Elastic Net chỉ chọn ở mức term riêng lẻ (term-level vs group-level selection)? [Limitations; Spec: FR-017](../spec.md#functional-requirements)

### 6. Deliverables & Tính Tái lập

- [x] **CHK022**: Notebook `notebooks/elastic_net_regression.ipynb` được cấu trúc đầy đủ, trình bày tiếng Việt, công thức LaTeX và tối thiểu 10 đồ thị phân tích có nhận xét định lượng? [Educational Value; Spec: FR-001, FR-020](../spec.md#functional-requirements)
- [x] **CHK023**: Cung cấp hàm `predict_student(raw_dict, model)` nhận đúng 12 raw features, trả về điểm GPA và các cảnh báo hợp lệ? [Application Interface; Spec: FR-018](../spec.md#functional-requirements)
- [x] **CHK024**: Xuất trained `Pipeline` hoàn chỉnh dạng file `.joblib` nhận 12 raw features, có thể nạp lại và tái lập dự đoán chính xác? [Serialization; Spec: FR-021](../spec.md#functional-requirements)
- [x] **CHK025**: Toàn bộ artifacts đầu ra (executed notebook, bảng CSV, hình ảnh PNG, file metrics JSON) được lưu trữ đầy đủ trong `outputs/elastic_net/`? [Deliverable Standards; Spec: FR-022](../spec.md#functional-requirements)
- [x] **CHK026**: Mã nguồn tái sử dụng được phân rã khoa học trong `src/lab01_ml/` theo đúng cấu trúc hiện có của repo? [Modularity; Spec: FR-023](../spec.md#functional-requirements)
- [x] **CHK027**: Bộ unit tests trong `tests/` bao phủ toàn diện: shape tests (39/40/54 terms), fold-only scaling, unknown category handling, serialization, và inference checks? [Quality Assurance; Spec: FR-024](../spec.md#functional-requirements)
- [x] **CHK028**: Toàn bộ quy trình có thể tái lập trên môi trường CPU chuẩn (Windows, Linux, Colab, Kaggle) mà không cần GPU hay mạng runtime? [Portability & Reproducibility; Spec: NFR-001, NFR-002](../spec.md#nonfunctional-requirements)
