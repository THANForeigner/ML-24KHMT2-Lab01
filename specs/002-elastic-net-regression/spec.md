# Feature Specification: Elastic Net Regression — Conditional GPA Prediction

**Feature Branch**: `002-elastic-net-regression`

**Created**: 2026-11-24

**Status**: Draft

**Input**: Xây dựng mô hình **Elastic Net Regression** để dự đoán `Post_Semester_GPA` (Conditional GPA Prediction / What-if Analysis) trên bộ dữ liệu synthetic *AI Impact on Students*, đồng bộ với quy ước của repo và mô hình `001-interaction-linear-regression`.

---

## Clarifications

### Session 2026-11-24

- **Quy tắc phân vùng & bảo vệ test set**: Tái sử dụng nguyên vẹn bộ dữ liệu đã chia sẵn 70/15/15 (`data/splits/` hoặc `data/{train,validation,test}/dataset.csv`), đối chiếu tính toàn vẹn với `data/assignments.csv` và `data/metadata.json`. Tuyệt đối không tự chia lại, không ghi đè dữ liệu. Chưa đọc target của tập `test` khi chưa có phê duyệt split chung từ nhóm; quy trình tuning chỉ chạy 5-fold Cross-Validation trên tập `train` và đánh giá trên tập `validation`, ghi nhận `final test: not executed` tương tự `001-interaction-linear-regression`.
- **Đóng gói tiền xử lý & Scaling**: Do Elastic Net áp dụng hàm phạt $L_1$ và $L_2$ đồng thời trên các hệ số hồi quy:
  $$\min_{\beta} \frac{1}{2n} \|y - X\beta\|_2^2 + \alpha \cdot l_1\_ratio \|\beta\|_1 + \frac{1}{2}\alpha(1 - l_1\_ratio)\|\beta\|_2^2$$
  toàn bộ biến số sau mã hóa và tương tác bắt buộc phải được chuẩn hóa (`StandardScaler`) để tránh việc biến có thang đo lớn bị phạt bất công. Toàn bộ các bước: imputation (median/mode), One-Hot Encoding drop-first, centering biến liên tục, tạo đa thức/tương tác, và `StandardScaler` MUST được đóng gói trọn vẹn trong `Pipeline` của `scikit-learn` và fit duy nhất trên training fold tương ứng (Tuyệt đối không rò rỉ thông tin từ validation/test).
- **Hệ phân cấp Feature Sets (EN1, EN2, EN3)**:
  - **EN1 (39 terms)**: 12 biến gốc sau One-Hot Encoding (5 biến numeric/binary + 34 dummy columns từ 7 biến categorical).
  - **EN2 (40–41 terms)**: `EN1` + Polynomial phi tuyến ($H_c = H - \overline{H}_{train}$, $H_c^2$; tùy chọn $P_c, P_c^2$). **Quy tắc bắt buộc**: Không giữ đồng thời biến thô $H$ và biến trừ tâm $H_c$ khi mô hình có hệ số chặn (intercept) để tránh cộng tuyến hoàn hảo.
  - **EN3 (54–55 terms — Trọng tâm nghiên cứu)**: `EN2` + 3 nhóm biến tương tác 2 chiều ($H_c \times \text{Primary\_Use\_Case dummy}$, $H_c \times \text{Prompt\_Engineering\_Skill dummy}$, $\text{Primary\_Use\_Case dummy} \times \text{Prompt\_Engineering\_Skill dummy}$), duy trì đầy đủ các thành phần bậc thấp (main effects).
- **Tối ưu hóa siêu tham số (Hyperparameter Tuning)**:
  - Tinh chỉnh đồng thời cặp siêu tham số $\alpha > 0$ (cường độ phạt tổng thể) và $l_1\_ratio \in [0, 1]$ (tỷ trọng kết hợp giữa Lasso $L_1$ và Ridge $L_2$) bằng **5-fold CV** trên tập `train` cho cả 3 cấu hình `EN1`, `EN2`, `EN3`.
  - Quy tắc hòa (tie-break): Nếu hai mô hình có CV RMSE chênh lệch trong ngưỡng $\le 0.1\%$ ($1.001 \times \min$), ưu tiên mô hình có độ thưa cao hơn (ít hệ số khác 0 hơn / ít predictor terms hơn).
- **Group Ablation (No-Anxiety)**: Huấn luyện và đánh giá thêm biến thể loại bỏ `Anxiety_Level_During_Exams` để đánh giá rủi ro rò rỉ thời gian (temporal leakage) khi dự đoán GPA trước kỳ thi.
- **Đặc thù trích xuất của Elastic Net**:
  - Trích xuất số hệ số khác 0 ($k = \|\hat{\beta}\|_0$), độ thưa (sparsity $\%= 1 - k/p$), và tần suất chọn biến (selection frequency) qua 5 fold huấn luyện để xác định tính ổn định của các biến được giữ lại.
  - Kiểm tra tính phân cấp (hierarchy check): ghi nhận liệu Elastic Net có tự ý giữ biến tương tác nhưng triệt tiêu biến thành phần (main effect) tương ứng hay không.
- **Giới hạn & Cảnh báo GPA**:
  - Ghi nhận tỷ lệ và số lượng dự đoán vượt trần GPA $[0.0, 4.0]$ (tuyệt đối không tự ý cắt/clip âm thầm để che giấu sai số phân phối).
  - Khẳng định tính chất synthetic của bộ dữ liệu (không suy luận quan hệ nhân quả — association $\neq$ causation).
  - Nhận định giới hạn của Elastic Net tiêu chuẩn: áp dụng phạt co cụm trên từng cột mã hóa riêng lẻ (term-level sparsity), không tự động loại toàn bộ nhóm biến phân loại (không phải Group Elastic Net).

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Hiểu bài toán & Cơ chế Elastic Net Regularization (Priority: P1)

Người học đọc notebook từ đầu để nắm vững bản chất toán học của Elastic Net Regression, sự cân bằng giữa co hệ số (Shrinkage - Ridge $L_2$) và chọn lọc đặc trưng (Sparsity - Lasso $L_1$), khắc phục nhược điểm của Lasso khi các đặc trưng có tương quan cao (grouping effect).

**Why this priority**: Cung cấp nền tảng lý thuyết vững chắc giúp người học hiểu rõ tại sao chọn Elastic Net cho bài toán dự đoán GPA có nhiều biến tương tác và đa cộng tuyến.

**Independent Test**: Đọc phần giải thích công thức toán học, hàm mất mát (loss function), và ví dụ minh họa bằng tay về cập nhật soft-thresholding có thêm điều kiện co $L_2$.

**Acceptance Scenarios**:
1. **Given** notebook được mở, **When** người học đọc phần giới thiệu lý thuyết, **Then** thấy rõ công thức hàm mục tiêu Elastic Net, ý nghĩa của $\alpha$ và $l_1\_ratio$, giải thích 12 biến đầu vào (nhóm A–D), 1 biến mục tiêu `Post_Semester_GPA`, và danh sách các cột bị cấm tuyệt đối.
2. **Given** bảng so sánh OLS vs Ridge vs Lasso vs Elastic Net, **When** đối chiếu, **Then** người học phân biệt được hành vi xử lý đa cộng tuyến và khả năng chọn biến của từng phương pháp.

---

### User Story 2 — Kiểm tra Dữ liệu & Ngăn ngừa Leakage (Priority: P1)

Hệ thống nạp dữ liệu từ phân vùng đóng băng (frozen splits), kiểm tra mã băm SHA-256, xác minh tính rời rạc của các tập (disjointness) và kiểm soát rò rỉ thông tin trước khi thực hiện bất kỳ bước huấn luyện nào.

**Why this priority**: Bảo vệ tính liêm chính học thuật và bảo đảm kết quả đánh giá là trung thực, không bị rò rỉ dữ liệu (data leakage).

**Independent Test**: Chạy kiểm tra tính toàn vẹn của split với `data/assignments.csv` và `data/metadata.json`, đảm bảo tập test không bị truy cập target.

**Acceptance Scenarios**:
1. **Given** các file chia sẵn trong `data/`, **When** pipeline nạp dữ liệu, **Then** xác minh đúng 35.000 dòng train, 7.500 dòng validation, 7.500 dòng test, không trùng lặp `Student_ID` giữa các tập, và mã băm khớp với `data/metadata.json`.
2. **Given** tập test trong suốt giai đoạn phát triển mô hình, **When** chạy notebook hoặc kiểm thử, **Then** target của tập test không bao giờ được nạp vào bộ nhớ hoặc tính toán; báo cáo ghi rõ `final test: not executed`.

---

### User Story 3 — Xây dựng Pipeline & Feature Sets EN1–EN3 (Priority: P1)

Hệ thống đóng gói toàn bộ quy trình tiền xử lý, mở rộng đặc trưng thành 3 tập `EN1`, `EN2`, `EN3`, và chuẩn hóa `StandardScaler` trong từng fold Cross-Validation.

**Why this priority**: Đảm bảo các đặc trưng được tạo tự động, không rò rỉ tham số thống kê (mean, median, standard deviation) giữa các tập dữ liệu, và kiểm soát số chiều chính xác.

**Independent Test**: Khởi tạo pipeline trên tập train và kiểm tra số lượng feature terms đầu ra của từng tập đặc trưng.

**Acceptance Scenarios**:
1. **Given** 12 raw features, **When** fit transformer `EN1`, **Then** tạo đúng 39 terms sau One-Hot Encoding drop-first (bao gồm 5 numeric/binary và 34 dummies cho các categorical).
2. **Given** transformer `EN2`, **When** bổ sung thành phần phi tuyến, **Then** tính $H_c = H - \overline{H}_{train}$ và $H_c^2$ (tùy chọn $P_c, P_c^2$), loại bỏ biến thô $H$ khỏi design matrix khi có intercept, tạo ra đúng 40–41 terms.
3. **Given** transformer `EN3`, **When** mở rộng các tương tác 2 chiều, **Then** thêm đầy đủ 14 interaction terms ($4 \text{ terms } H_c \times U_i$, $2 \text{ terms } H_c \times K_j$, $8 \text{ terms } U_i \times K_j$), đạt đúng 54–55 terms và giữ trọn vẹn các main effects.
4. **Given** một danh mục chưa từng xuất hiện (unknown category) trong validation hoặc inference, **When** biến đổi qua pipeline, **Then** được mã hóa an toàn thành vector 0 kèm warning cảnh báo người dùng.

---

### User Story 4 — Tinh chỉnh Siêu tham số & Huấn luyện (Priority: P1)

Thực hiện 5-fold Cross-Validation trên tập `train` để tìm cặp siêu tham số tối ưu $(\alpha^*, l_1\_ratio^*)$ cho từng bộ đặc trưng `EN1`, `EN2`, `EN3`.

**Why this priority**: Tìm kiếm sự cân bằng tối ưu giữa bias và variance, đồng thời xác định mức độ co và độ thưa thích hợp nhất cho bài toán.

**Independent Test**: Chạy grid search hoặc coordinate descent path trên 5 folds, đối chiếu bảng kết quả siêu tham số với tiêu chí RMSE nhỏ nhất.

**Acceptance Scenarios**:
1. **Given** không gian tìm kiếm $\alpha \in [10^{-4}, 10^1]$ và $l_1\_ratio \in [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]$, **When** chạy 5-fold CV trên tập train, **Then** ghi nhận đầy đủ MAE, MSE, RMSE, $R^2$ trung bình và độ lệch chuẩn ($\pm \text{sd}$) cho từng cặp siêu tham số.
2. **Given** các mô hình có CV RMSE chênh lệch trong ngưỡng $\le 0.1\%$, **When** áp dụng tie-break policy, **Then** tự động ưu tiên mô hình có số hệ số khác 0 ít hơn (độ thưa cao hơn); nếu vẫn hòa thì ưu tiên mô hình ít terms tổng thể hơn.
3. **Given** mô hình tối ưu đã chọn, **When** kết thúc giai đoạn tuning trên train, **Then** fit lại pipeline hoàn chỉnh với $(\alpha^*, l_1\_ratio^*)$ duy nhất trên toàn bộ tập train (không đụng đến validation để fit).

---

### User Story 5 — Group Ablation (Loại bỏ biến Temporal Leakage) (Priority: P1)

Thực hiện phân tích độ nhạy (sensitivity analysis) bằng cách loại bỏ biến `Anxiety_Level_During_Exams` để tạo biến thể mô hình No-Anxiety áp dụng cho kịch bản dự đoán trước kỳ thi.

**Why this priority**: `Anxiety_Level_During_Exams` xảy ra trong kỳ thi, nếu dùng để dự đoán trước kỳ thi sẽ gây rò rỉ thời gian (temporal leakage).

**Independent Test**: Huấn luyện biến thể No-Anxiety trên cùng 5-fold CV và so sánh trực tiếp với mô hình Full trên tập validation.

**Acceptance Scenarios**:
1. **Given** tập dữ liệu không chứa cột `Anxiety_Level_During_Exams` (11 raw features), **When** chạy pipeline `EN3_no_anxiety`, **Then** tạo đúng 45–46 terms (do bớt 9 dummy levels của Anxiety).
2. **Given** kết quả đánh giá trên tập validation, **When** so sánh Full vs No-Anxiety, **Then** báo cáo rõ mức độ thay đổi của RMSE, MAE, $R^2$, và chỉ ra sự đánh đổi giữa độ chính xác dự đoán và tính thực tiễn khi triển khai.

---

### User Story 6 — Đánh giá Validation & So sánh OLS Baseline (Priority: P1)

Đánh giá các mô hình Elastic Net đã huấn luyện trên tập `validation` độc lập, so sánh trực tiếp với OLS baseline từ mô hình `001-interaction-linear-regression`.

**Why this priority**: Xác nhận xem việc áp dụng hiệu chỉnh phạt (regularization) có thực sự cải thiện khả năng tổng quát hóa so với hồi quy tuyến tính cổ điển hay không.

**Independent Test**: Nạp tập `validation`, tính toán các chỉ số dự báo và đối chiếu bảng so sánh giữa OLS và Elastic Net (`EN1`, `EN2`, `EN3`).

**Acceptance Scenarios**:
1. **Given** tập `validation` (7.500 mẫu), **When** dự đoán bằng các mô hình đã fit trên tập `train`, **Then** hiển thị đầy đủ bảng so sánh gồm RMSE, MAE, MSE, $R^2$ cho OLS baseline và Elastic Net (`EN1`, `EN2`, `EN3`).
2. **Given** hiện tượng đa cộng tuyến trong `EN3`, **When** so sánh với OLS `I3`, **Then** phân tích xem Elastic Net có giảm phương sai của hệ số (variance reduction) và cải thiện sai số ngoại suy hay không.

---

### User Story 7 — Phân tích Đặc thù Elastic Net: Sparsity, Selection Frequency & Hierarchy (Priority: P1)

Trích xuất các chỉ số nội tại chuyên sâu của Elastic Net để hiểu rõ cơ chế co hệ số và chọn lọc biến.

**Why this priority**: Elastic Net không chỉ là công cụ dự báo mà còn là công cụ khám phá cấu trúc dữ liệu; việc phân tích các thuộc tính này chứng minh tính nghiêm ngặt khoa học của nghiên cứu.

**Independent Test**: Truy xuất thuộc tính `coef_` của mô hình tối ưu và thống kê các hệ số qua 5 fold CV.

**Acceptance Scenarios**:
1. **Given** mô hình Elastic Net tối ưu, **When** kiểm tra `coef_`, **Then** thống kê chính xác số lượng hệ số khác 0 ($k$), số lượng hệ số bị triệt tiêu về 0, và tỷ lệ thưa (sparsity rate $\%= (1 - k/p) \times 100$).
2. **Given** 5 fold huấn luyện của CV, **When** đếm tần suất xuất hiện hệ số khác 0 của từng đặc trưng, **Then** lập bảng và biểu đồ tần suất chọn biến ($0/5, 1/5, \dots, 5/5$), chỉ ra những biến có tính ổn định cao nhất (core predictors).
3. **Given** các cặp tương tác $H_c \times U_i$ và $U_i \times K_j$, **When** kiểm tra tính phân cấp (hierarchy check), **Then** chỉ ra cụ thể các trường hợp biến tương tác được chọn nhưng biến chính (main effect) bị loại bỏ hoặc ngược lại, nêu rõ hạn chế của Elastic Net tiêu chuẩn.

---

### User Story 8 — Kiểm soát Trần GPA & Nhận diện Giới hạn Nghiên cứu (Priority: P1)

Hệ thống theo dõi các dự đoán ngoài phạm vi hợp lệ $[0.0, 4.0]$ và nêu rõ các giới hạn phương pháp luận.

**Why this priority**: Tránh ngộ nhận về mặt định lượng, không bóp méo phân phối dự đoán và ngăn chặn việc suy luận nhân quả sai lệch.

**Independent Test**: Quét toàn bộ vector dự đoán trên tập validation để đếm số lượng $\hat{y} < 0.0$ hoặc $\hat{y} > 4.0$.

**Acceptance Scenarios**:
1. **Given** kết quả dự đoán trên tập validation, **When** kiểm tra cận trên/dưới, **Then** báo cáo số lượng và tỷ lệ phần trăm dự đoán vượt trần GPA ($> 4.0$) và dưới đáy ($< 0.0$); tuyệt đối không tự ý áp dụng hàm `clip()` âm thầm trước khi tính metrics.
2. **Given** kết quả phân tích hệ số và đường cong phản hồi what-if, **When** diễn giải ý nghĩa, **Then** tài liệu và notebook phải khẳng định rõ: dữ liệu là synthetic, liên hệ tìm được là tương quan có điều kiện (conditional association), không suy diễn tác động nhân quả (causal impact).
3. **Given** bản chất của Elastic Net, **When** đánh giá khả năng chọn đặc trưng, **Then** chỉ rõ Elastic Net hoạt động ở mức từng dummy riêng lẻ (term-level), có thể giữ lại một vài dummy của một biến phân loại và loại bỏ các dummy khác, không tương đương với việc chọn/loại bỏ nguyên cả biến danh mục (group selection).

---

### User Story 9 — Tích hợp Pipeline & Dự đoán Kịch bản What-if (Priority: P1)

Xuất mô hình ra tệp nhị phân `.joblib` kèm hàm giao diện `predict_student(raw_dict, model)` nhận đầu vào là đúng 12 raw features, cho phép phân tích kịch bản What-if khi thay đổi số giờ dùng AI.

**Why this priority**: Cung cấp sản phẩm hoàn chỉnh có thể tái sử dụng cho các ứng dụng thực tế hoặc giao diện người dùng.

**Independent Test**: Nạp một dictionary 12 trường raw data mẫu, thực hiện dự đoán và mô phỏng thay đổi `Weekly_GenAI_Hours` từ 0 đến 40 giờ.

**Acceptance Scenarios**:
1. **Given** một dictionary hoặc pandas DataFrame gồm đúng 12 raw features, **When** gọi `predict_student(raw_dict, pipeline)`, **Then** trả về điểm GPA dự đoán, tên mô hình (`EN3`), kịch bản (`full` hoặc `no_anxiety`), và danh sách cảnh báo nếu có giá trị ngoài miền hoặc danh mục chưa biết.
2. **Given** kịch bản What-if, **When** cố định 11 biến nền và quét `Weekly_GenAI_Hours` từ 0 đến 40 giờ, **Then** vẽ đồ thị đường cong đáp ứng GPA dự đoán kèm chú thích rõ ràng rằng đây là giả lập có điều kiện, không phải can thiệp thực nghiệm.
3. **Given** pipeline lưu trong file `.joblib`, **When** nạp lại ở môi trường sạch, **Then** thực hiện dự đoán hoàn toàn trùng khớp với kết quả ban đầu (dung sai sai số số thực $\le 10^{-9}$).

---

### User Story 10 — Bàn giao & Tái lập Nghiệm thu (Priority: P1)

Tất cả các thành phần (notebook, mã nguồn package, unit tests, outputs) được tổ chức khoa học, có thể tái lập 100% bằng cách chạy script hoặc Restart Kernel & Run All trên môi trường CPU tiêu chuẩn.

**Why this priority**: Đảm bảo toàn bộ deliverables của Lab 01 đáp ứng tiêu chuẩn phần mềm chuyên nghiệp và yêu cầu chấm điểm của môn học.

**Independent Test**: Chạy toàn bộ test suite bằng `pytest` và chạy notebook qua clean kernel runner.

**Acceptance Scenarios**:
1. **Given** môi trường đã cài `requirements.txt`, **When** chạy `pytest tests/`, **Then** toàn bộ unit tests liên quan đến feature engineering, data split guard, model training, sparsity checks và inference đều vượt qua.
2. **Given** thư mục `outputs/elastic_net/`, **When** kiểm tra sau khi chạy, **Then** có đầy đủ notebook đã execute, các bảng số liệu CSV, các đồ thị PNG độ phân giải cao, file số liệu `metrics.json`, và mô hình `selected_pipeline.joblib`.

---

### Edge Cases

- **Tất cả các hệ số bị triệt tiêu về 0**: Khi $\alpha$ quá lớn, toàn bộ hệ số hồi quy bị triệt tiêu về 0, mô hình chỉ còn lại intercept $\hat{y} = \overline{y}_{train}$. Pipeline phải xử lý gracefully, cảnh báo độ thưa đạt 100%, và không gây lỗi chia cho 0 khi tính toán metrics.
- **Giá trị $l_1\_ratio = 1$ (Lasso thuần) hoặc $l_1\_ratio = 0$ (Ridge thuần)**: Pipeline phải hỗ trợ trơn tru các trường hợp biên của Elastic Net mà không phát sinh lỗi cảnh báo không hội tụ (convergence warnings) bằng cách điều chỉnh số vòng lặp tối đa `max_iter` và dung sai `tol`.
- **Cảnh báo không hội tụ (Convergence Warning)**: Với dữ liệu nhiều chiều và $\alpha$ nhỏ, thuật toán Coordinate Descent có thể không hội tụ kịp trong 1.000 bước. Thiết lập `max_iter=10000` và `tol=1e-4` để đảm bảo nghiệm tối ưu đạt độ chính xác cao.
- **Dữ liệu thiếu cấp độ danh mục (Missing category levels in train folds)**: Trong từng fold của 5-fold CV, nếu một fold ngẫu nhiên thiếu một level danh mục hiếm, OneHotEncoder phải xác định số cột thực tế, cảnh báo rõ và không gây lệch chiều ma trận đầu vào của estimator.
- **Dữ liệu đầu vào suy biến hoặc giá trị vô hạn (`inf`, `-inf`, `NaN`)**: Hàm tiền xử lý phải phát hiện sớm, điền khuyết bằng train median/mode đã lưu hoặc ném ngoại lệ rõ ràng nêu đích danh cột bị lỗi.
- **Dự đoán ngoài miền $[0.0, 4.0]$**: Không cắt/clip âm thầm; ghi nhận tỷ lệ out-of-bounds vào báo cáo đánh giá chất lượng mô hình.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001 — Ngôn ngữ & Trình bày**: Notebook `notebooks/elastic_net_regression.ipynb` MUST trình bày bằng tiếng Việt, chuẩn LaTeX cho mọi công thức toán học, cấu trúc theo thứ tự logic: Lý thuyết & Động cơ $\to$ Data Integrity & Schema $\to$ Pipeline & Feature Sets $\to$ 5-Fold CV Tuning $\to$ Group Ablation $\to$ So sánh Baseline OLS $\to$ Phân tích Đặc thù (Sparsity, Frequency, Hierarchy) $\to$ Giới hạn & Kiểm tra Trần GPA $\to$ Demo What-if & Tích hợp ứng dụng.
- **FR-002 — Hợp đồng Dữ liệu Đầu vào (X)**: Đầu vào thô của mô hình MUST gồm đúng 12 biến thuộc các nhóm A–D:
  - Nhóm A (Academic): `Pre_Semester_GPA` (float), `Traditional_Study_Hours` (float), `Year_of_Study` (categorical), `Major_Category` (categorical).
  - Nhóm B (AI Usage): `Weekly_GenAI_Hours` (float), `Primary_Use_Case` (categorical), `Prompt_Engineering_Skill` (categorical), `Tool_Diversity` (numeric), `Paid_Subscription` (binary/bool).
  - Nhóm C (Contextual): `Institutional_Policy` (categorical).
  - Nhóm D (Psychological): `Perceived_AI_Dependency` (categorical, 1–10 levels), `Anxiety_Level_During_Exams` (categorical, 1–10 levels).
- **FR-003 — Biến Mục tiêu & Danh sách Loại trừ Tuyệt đối**: Target duy nhất là `Post_Semester_GPA`. Bắt buộc loại trừ tuyệt đối các cột: `Student_ID`, `GPA_Change`, `Skill_Retention_Score`, `Burnout_Risk_Level` khỏi không gian đặc trưng $X$ để ngăn ngừa rò rỉ dữ liệu (target/outcome leakage).
- **FR-004 — Phân vùng Dữ liệu & Đóng băng Phân chia**: Pipeline MUST nạp trực tiếp bộ chia 70/15/15 đã đóng băng trong repo (35.000 train / 7.500 validation / 7.500 test), kiểm tra tính toàn vẹn với `data/assignments.csv` và `data/metadata.json`. Tuyệt đối không tự ý chia lại hoặc ghi đè file split.
- **FR-005 — Bảo vệ Tập Holdout (Test Guard)**: Trong suốt quá trình phát triển mô hình và điều chỉnh siêu tham số, tuyệt đối KHÔNG đọc cột mục tiêu của tập `test`. Mọi so sánh chỉ thực hiện trên 5-fold CV của tập `train` và đánh giá trên tập `validation`. Trạng thái của test set được ghi nhận rõ là `final test: not executed` cho đến khi có sự thống nhất chung của toàn nhóm.
- **FR-006 — Đóng gói Pipeline & Chống Leakage Tiền xử lý**: Mọi thao tác học tham số tiền xử lý: điền giá trị thiếu (median/mode), mã hóa One-Hot Encoding (`drop='first'`, `handle_unknown='ignore'`), tính giá trị trung bình để trừ tâm ($H_c = H - \overline{H}_{train}$), và chuẩn hóa thang đo (`StandardScaler`) MUST được đóng gói bên trong `sklearn.pipeline.Pipeline` và chỉ học trên training set/fold tương ứng.
- **FR-007 — Định nghĩa Feature Set EN1 (39 terms)**:
  - Gồm 12 biến gốc sau One-Hot Encoding bỏ mức tham chiếu (`drop='first'`).
  - 5 biến số/nhị phân: `Weekly_GenAI_Hours`, `Pre_Semester_GPA`, `Traditional_Study_Hours`, `Tool_Diversity`, `Paid_Subscription`.
  - 34 dummy variables: `Year_of_Study` (4 dummies), `Major_Category` (4 dummies), `Primary_Use_Case` (4 dummies), `Prompt_Engineering_Skill` (2 dummies), `Institutional_Policy` (2 dummies), `Perceived_AI_Dependency` (9 dummies), `Anxiety_Level_During_Exams` (9 dummies).
  - Tổng cộng đúng **39 predictor terms** (chưa tính hệ số chặn).
- **FR-008 — Định nghĩa Feature Set EN2 (40–41 terms)**:
  - Mở rộng từ `EN1` bằng cách bổ sung thành phần phi tuyến: $H_c = H - \overline{H}_{train}$ và $H_c^2$ (tùy chọn $P_c, P_c^2$).
  - **Quy tắc bảo toàn độc lập tuyến tính**: Khi mô hình có intercept, KHÔNG giữ đồng thời biến thô $H$ và biến trừ tâm $H_c$.
  - Tổng số predictor terms: đúng **40 terms** (nếu chỉ thêm $H_c^2$) hoặc **41 terms** (nếu thêm cả $H_c^2$ và $P_c^2$).
- **FR-009 — Định nghĩa Feature Set EN3 (54–55 terms — Trọng tâm nghiên cứu)**:
  - Mở rộng từ `EN2` bằng cách bổ sung 3 nhóm biến tương tác 2 chiều:
    1. Tương tác AI Hours $\times$ Use Case: $H_c \times U_i$ ($i = 1 \dots 4$, 4 terms).
    2. Tương tác AI Hours $\times$ Prompt Skill: $H_c \times K_j$ ($j = 1, 2$, 2 terms).
    3. Tương tác Use Case $\times$ Prompt Skill: $U_i \times K_j$ ($i = 1 \dots 4, j = 1, 2$, 8 terms).
  - Tổng số interaction terms bổ sung: $4 + 2 + 8 = 14$ terms.
  - Tổng số predictor terms: đúng **54 terms** (hoặc **55 terms** nếu có $P_c^2$), duy trì trọn vẹn tất cả các main effects.
- **FR-010 — Chuẩn hóa Đặc trưng (StandardScaler)**:
  - Do Elastic Net phạt tổng độ lớn hệ số ($L_1 + L_2$), mọi predictor terms (cả biến liên tục, biến nhị phân, biến dummy và biến tương tác) MUST được chuẩn hóa về cùng thang đo (trung bình 0, phương sai 1) qua `StandardScaler(with_mean=True, with_std=True)` bên trong Pipeline trước khi đưa vào bộ ước lượng `ElasticNet`.
- **FR-011 — Tinh chỉnh Siêu tham số qua 5-Fold Cross-Validation**:
  - Thực hiện tìm kiếm lưới (Grid Search) trên tập `train` cho cả 3 không gian đặc trưng (`EN1`, `EN2`, `EN3`).
  - Lưới tham số MUST bao phủ:
    - $\alpha \in [10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0]$ (hoặc dải log-spaced phù hợp).
    - $l_1\_ratio \in [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]$.
  - Đánh giá bằng 5-fold CV với seed cố định (`random_state=42`).
- **FR-012 — Quy tắc Chọn Mô hình & Tie-Break**:
  - Tiêu chuẩn chính: mô hình có mean CV RMSE nhỏ nhất trên 5 folds.
  - Tiêu chuẩn hòa (tie-break): các ứng viên có mean CV RMSE trong vòng $0.1\%$ ($1.001 \times \text{min\_RMSE}$) được coi là tương đương về hiệu năng. Ưu tiên chọn mô hình có số hệ số khác 0 ít hơn (độ thưa lớn hơn để tăng tính tổng quát); nếu vẫn hòa thì chọn mô hình có cấu hình đặc trưng đơn giản hơn (`EN1` < `EN2` < `EN3`).
- **FR-013 — Phân tích Group Ablation (No-Anxiety)**:
  - Huấn luyện cấu hình `EN3_no_anxiety` loại bỏ biến `Anxiety_Level_During_Exams` (còn 11 raw features, tạo 45–46 terms).
  - Đánh giá trên 5-fold CV và tập validation để định lượng chi phí suy giảm độ chính xác khi chuyển sang kịch bản dự đoán trước kỳ thi (pre-exam forecasting).
- **FR-014 — Đánh giá Validation & So sánh OLS Baseline**:
  - Đánh giá hiệu năng trên tập `validation` (7.500 mẫu) cho các mô hình: OLS baseline (`I0`, `I1`, `I2`, `I3`), Elastic Net (`EN1`, `EN2`, `EN3`), và biến thể Ablation (`EN3_no_anxiety`).
  - Báo cáo chi tiết 4 chỉ số: RMSE, MAE, MSE, $R^2$.
- **FR-015 — Trích xuất Chỉ số Đặc thù Elastic Net**:
  - **Sparsity**: Đếm số hệ số khác 0 ($k$) và tính tỷ lệ thưa ($\% = (1 - k/p) \times 100$).
  - **Selection Frequency**: Ghi nhận tần suất một đặc trưng được chọn (hệ số $\neq 0$) qua 5 fold của CV ($0/5 \dots 5/5$), lập biểu đồ cột thể hiện mức độ bền vững (stability) của các đặc trưng.
  - **Hierarchy Check**: Kiểm tra và báo cáo các trường hợp vi phạm tính phân cấp (ví dụ: biến tương tác $H_c \times U_i$ được chọn nhưng biến $H_c$ hoặc $U_i$ bị phạt về 0).
- **FR-016 — Theo dõi Dự đoán Ngoài Miền (GPA Ceiling & Floor)**:
  - Thống kê số lượng và tỷ lệ phần trăm dự đoán $\hat{y} > 4.0$ và $\hat{y} < 0.0$ trên tập validation.
  - Tuyệt đối không tự ý dùng hàm cắt `np.clip(pred, 0, 4)` trước khi tính toán các chỉ số sai số; ghi nhận hiện tượng này như một bằng chứng về giới hạn mô hình tuyến tính có điều kiện.
- **FR-017 — Ghi nhận Giới hạn Nghiên cứu**:
  - Bắt buộc có phần thảo luận giới hạn:
    1. Dữ liệu là synthetic, mọi kết luận chỉ là quan sát tương quan có điều kiện (conditional association), không suy diễn quan hệ nhân quả.
    2. Kịch bản dự đoán mang tính hồi cứu (retrospective), thời điểm thu thập dữ liệu trong học kỳ chưa được xác nhận độc lập.
    3. Elastic Net chuẩn chỉ lọc từng term riêng lẻ (term-level sparsity), không bảo đảm tính toàn vẹn nhóm phân loại (group selection).
- **FR-018 — Interface Dự đoán & Fallback**:
  - Cung cấp hàm `predict_student(raw_dict, model)` nhận dictionary 12 raw features, trả về dict kết quả: `prediction` (float), `model` (`EN3`), `scenario` (`full` / `no_anxiety`), `warnings` (danh sách cảnh báo).
  - Kiểm tra tính hợp lệ của đầu vào: thiếu trường dữ liệu, sai kiểu, giá trị âm ngoài miền logic, hoặc danh mục chưa thấy.
- **FR-019 — Trực quan hóa Phân tích What-if**:
  - Tạo biểu đồ tương tác What-if: giữ nguyên 11 biến nền ở một hồ sơ sinh viên tiêu chuẩn (representative profile) và mô phỏng sự biến thiên của `Weekly_GenAI_Hours` từ 0 đến 40 giờ trên các nhóm Use Case và Prompt Skill khác nhau.
- **FR-020 — Biểu đồ Đánh giá (Tối thiểu 10 hình ảnh phân tích)**:
  - Notebook MUST tạo tối thiểu 10 biểu đồ trực quan hóa độc lập:
    1. Phân phối target GPA trên tập train.
    2. Phân phối thời gian dùng AI (`Weekly_GenAI_Hours`).
    3. Đường cong nghiệm (Regularization Path) của Elastic Net khi $\alpha$ thay đổi.
    4. Bề mặt sai số (Validation Heatmap) theo lưới cặp $(\alpha, l_1\_ratio)$.
    5. So sánh CV RMSE giữa các mô hình (`EN1`, `EN2`, `EN3` vs OLS).
    6. So sánh Validation RMSE và $R^2$ giữa các mô hình.
    7. Tần suất chọn biến (Selection Frequency) qua 5 folds.
    8. Độ thưa (Sparsity) và độ lớn hệ số hồi quy được giữ lại.
    9. Phân tích phần dư (Residuals vs Fitted) trên tập validation.
    10. Biểu đồ What-if Response Curves theo giờ AI cho các nhóm Use Case.
  - Mỗi đồ thị bắt buộc có title, labels, nguồn tập dữ liệu và 1–3 câu nhận xét dựa trên số liệu thực tế.
- **FR-021 — Xuất Pipeline Huấn luyện (.joblib)**:
  - Xuất mô hình tối ưu đã fit trên toàn bộ tập `train` thành file `outputs/elastic_net/models/selected_elastic_net_pipeline.joblib`.
  - File model này phải nhận trực tiếp DataFrame 12 raw features và tự động thực hiện trọn vẹn tiền xử lý $\to$ scaling $\to$ dự đoán.
- **FR-022 — Lưu trữ Kết quả & Tái lập (Artifacts)**:
  - Toàn bộ kết quả đầu ra MUST được lưu trong `outputs/elastic_net/`:
    - `executed_notebook.ipynb`: notebook đã chạy hoàn chỉnh từ clean kernel.
    - `tables/`: bảng so sánh mô hình, kết quả 5-fold CV, bảng tần suất chọn biến dưới dạng CSV.
    - `figures/`: các biểu đồ PNG độ phân giải cao (300 DPI).
    - `metrics.json`: lưu thông tin siêu tham số tối ưu, CV scores, validation scores, độ thưa, và tỷ lệ ngoài miền.
- **FR-023 — Cấu trúc Module Tái sử dụng**:
  - Mã nguồn tái sử dụng được đặt trong `src/lab01_ml/`:
    - `features/elastic_net_features.py`: Transformer `ElasticNetFeatures` kế thừa `BaseEstimator`, `TransformerMixin`.
    - `models/elastic_net_training.py`: Hàm huấn luyện, tuning 5-fold CV, group ablation.
    - `models/elastic_net_evaluation.py`: Hàm trích xuất sparsity, selection frequency, hierarchy check, out-of-bounds check.
- **FR-024 — Unit Tests Toàn diện**:
  - Viết các bài kiểm thử tự động trong `tests/`:
    - Kiểm tra số chiều đầu ra của `EN1` (39), `EN2` (40–41), `EN3` (54–55) khi đủ levels.
    - Kiểm tra tính độc lập của scaling và centering (chỉ fit trên train rows).
    - Kiểm tra cơ chế xử lý unknown categories.
    - Kiểm tra hàm `predict_student` với dữ liệu thô hợp lệ, dữ liệu lỗi, và cảnh báo ngoài miền.
    - Kiểm tra việc nạp lại mô hình `.joblib` và tái lập dự đoán.

---

### Nonfunctional Requirements

- **NFR-001 — Khả năng Tái lập (Reproducibility)**: Khi chạy lại trên cùng môi trường với cùng cấu hình và seed (`random_state=42`), toàn bộ kết quả siêu tham số, metrics, và hệ số hồi quy phải tái lập chính xác 100% với dung sai số thực $\le 10^{-7}$.
- **NFR-002 — Tính Độc lập & Môi trường Thực thi (Portability)**: Toàn bộ quy trình phải chạy mượt mà trên môi trường CPU chuẩn (Windows, Linux, Colab, Kaggle) mà không đòi hỏi GPU hoặc kết nối mạng internet lúc runtime.
- **NFR-003 — Tính Minh bạch & Liêm chính Dữ liệu (Integrity)**: Tuyệt đối không hardcode kết quả, không tạo dữ liệu giả mạo; nếu thiếu dữ liệu phân vùng, hệ thống phải dừng với thông báo lỗi cụ thể, không tự ý chuyển trạng thái hoặc giả định kết quả.
- **NFR-004 — Tính Giáo dục & Tự giải thích (Self-contained Learning)**: Mã nguồn và notebook phải có giải thích chi tiết, chú thích rõ ràng, giúp người đọc nắm vững cả lý thuyết toán học lẫn kỹ thuật cài đặt chuẩn công nghiệp.

---

### Key Entities

- **Student Raw Record**: Bản ghi sinh viên gồm đúng 12 đặc trưng thuộc 4 nhóm A–D; target `Post_Semester_GPA` chỉ dùng lúc huấn luyện và đánh giá; `Student_ID` chỉ dùng kiểm tra phân vùng.
- **Feature Set Specification**: Định nghĩa tập biến `EN1` (39 terms), `EN2` (40–41 terms), `EN3` (54–55 terms) cùng thông tin mức tham chiếu và tâm biến số.
- **Elastic Net Hyperparameter State**: Cặp giá trị $(\alpha, l_1\_ratio)$, phạm vi lưới tìm kiếm, và thông tin tối ưu qua 5-fold CV.
- **Fold Evaluation Result**: Kết quả đánh giá trên từng fold gồm fold index, số lượng mẫu train/val, MAE, MSE, RMSE, $R^2$, số lượng hệ số $\neq 0$, và danh sách các biến được chọn.
- **Sparsity & Hierarchy Diagnostics**: Bảng thống kê tỷ lệ thưa, tần suất chọn biến qua 5 folds, và các cặp vi phạm quy tắc phân cấp.
- **Trained Pipeline Artifact**: Đối tượng đóng gói scikit-learn (`Pipeline`) chứa `ElasticNetFeatures` $\to$ `StandardScaler` $\to$ `ElasticNet`, được tuần tự hóa dạng `.joblib`.

---

### Out of Scope

- Tự động chia lại dữ liệu hoặc tạo bộ phân vùng mới ngoài bộ dữ liệu đã đóng băng trong repo.
- Đọc hoặc đánh giá trên target của tập `test` trước khi có quyết định phê duyệt bằng văn bản từ nhóm nghiên cứu.
- Các mô hình phi tuyến tính phức tạp như Deep Learning, Random Forest, GBDT, hoặc Causal Inference (Do-calculus).
- Regularization dạng Group Lasso (nhóm toàn bộ biến dummy) do nằm ngoài phạm vi Elastic Net tiêu chuẩn của scikit-learn.
- Triển khai ứng dụng Web production phức tạp; chỉ tập trung vào notebook giáo dục, module thư viện tái sử dụng và API inference cục bộ.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Xác minh đúng 35.000 dòng train, 7.500 dòng validation, 7.500 dòng test từ dữ liệu đã đóng băng; không có bất kỳ rò rỉ target test nào trong quá trình tuning.
- **SC-002**: Transformer tạo ra chính xác 39 terms cho `EN1`, 40 hoặc 41 terms cho `EN2`, và 54 hoặc 55 terms cho `EN3` khi kiểm tra trên dữ liệu đủ levels; duy trì đầy đủ main effects.
- **SC-003**: 100% các bước tính tham số trung bình, phương sai, median, mode và `StandardScaler` được thực hiện khép kín trong từng fold huấn luyện, không rò rỉ sang validation.
- **SC-004**: Hoàn thành 5-fold Cross-Validation trên tập train cho lưới siêu tham số $\alpha$ và $l_1\_ratio$, chọn ra cặp tối ưu theo tiêu chí RMSE nhỏ nhất và tie-break độ thưa.
- **SC-005**: Hoàn thành đánh giá mô hình tối ưu trên tập validation, trích xuất đầy đủ 4 chỉ số RMSE, MAE, MSE, $R^2$ và so sánh tường minh với OLS baseline.
- **SC-006**: Trích xuất chính xác số lượng hệ số khác 0 ($k$), tỷ lệ thưa ($\%$), bảng tần suất chọn biến qua 5 folds ($0/5 \dots 5/5$), và báo cáo kiểm tra tính phân cấp.
- **SC-007**: Ghi nhận chính xác số lượng và tỷ lệ phần trăm dự đoán vượt ngưỡng GPA $> 4.0$ hoặc $< 0.0$ mà không dùng hàm `clip()` âm thầm.
- **SC-008**: Xuất mô hình `.joblib` hoàn chỉnh; nạp lại và thực thi dự đoán trên 1 profile raw data mẫu thành công, trả về điểm số và cảnh báo hợp lệ.
- **SC-009**: Tạo ra tối thiểu 10 biểu đồ trực quan hóa độc lập trong notebook, mỗi biểu đồ có chú thích và nhận xét phân tích định lượng từ số liệu thực tế.
- **SC-010**: 100% unit tests trong `tests/` vượt qua thành công (`pytest` pass) và notebook chạy trơn tru từ đầu đến cuối trong môi trường sạch (`Restart Kernel & Run All`).

---

## Assumptions

- Phân vùng dữ liệu hiện tại trong `data/` (`assignments.csv`, `metadata.json`, và các file dataset) là nguồn chân lý duy nhất cho toàn bộ các thực nghiệm so sánh trong Lab 01.
- Mọi biến phân loại (`Year_of_Study`, `Major_Category`, `Primary_Use_Case`, `Prompt_Engineering_Skill`, `Institutional_Policy`, `Perceived_AI_Dependency`, `Anxiety_Level_During_Exams`) được mã hóa One-Hot với mức tham chiếu cố định (level đầu tiên theo thứ tự từ điển hoặc xuất hiện đầu tiên trong train) để đảm bảo tính diễn giải.
- Chuẩn hóa `StandardScaler` là điều kiện tiên quyết bắt buộc trước khi đưa ma trận đặc trưng vào bộ giải Coordinate Descent của Elastic Net.
- Tập test tiếp tục được giữ bí mật (holdout) và không được đánh giá trong spec này, chỉ ghi nhận `final test: not executed` để bảo vệ tính khách quan cho đến khi hoàn thiện toàn bộ báo cáo tổng kết của nhóm.

---

## Open Issues for `/speckit.clarify`

1. **Biến thể phi tuyến EN2**: Thống nhất xem `EN2` chỉ bổ sung $H_c^2$ (40 terms) hay bổ sung cả $P_c^2$ (41 terms) làm cấu hình mặc định cho các mô hình so sánh tiếp theo.
2. **Quy tắc hiển thị hệ số sau khi Scale**: Thống nhất việc báo cáo hệ số hồi quy: giữ nguyên ở thang đo đã chuẩn hóa (standardized coefficients) để so sánh tầm quan trọng tương đối, hay chuyển đổi ngược (unscale / back-transform) về thang đo gốc của biến.
3. **Thời điểm mở khóa tập Test**: Xác nhận mốc thời gian nhóm thống nhất mở khóa tập test và biến môi trường cho phép kích hoạt đánh giá holdout cuối cùng (`LAB_GROUP_SPLIT_APPROVED=yes`).
