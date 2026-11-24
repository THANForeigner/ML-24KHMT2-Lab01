# Research & Technical Decisions: Elastic Net Regression

**Ngày ghi nhận**: 2026-11-24  
**Phạm vi**: Khảo sát lý thuyết, căn cứ lựa chọn giải thuật, và các quyết định thiết kế mô hình Elastic Net Regression cho bài toán Conditional GPA Prediction.

---

## 1. Động cơ & Bản chất Toán học của Elastic Net

### Vấn đề của OLS, Ridge và Lasso thuần túy trong mô hình Interaction
- **Hồi quy OLS (Ordinary Least Squares)**: Khi mở rộng không gian đặc trưng với đa thức bậc hai ($H_c^2$) và các tương tác 2 chiều ($H_c \times U_i, H_c \times K_j, U_i \times K_j$), số chiều ma trận tăng vọt lên 54–55 terms. Giữa các biến này tồn tại mức độ tương quan đa cộng tuyến rất cao (multicollinearity), dẫn đến ma trận hiệp phương sai $(X^TX)$ có định thức gần 0 (ill-conditioned), phương sai của các ước lượng hệ số $\text{Var}(\hat{\beta})$ bùng nổ, làm giảm nghiêm trọng khả năng tổng quát hóa trên tập dữ liệu mới.
- **Ridge Regression ($L_2$ Penalty)**: Co các hệ số về gần 0 bằng cách cộng lượng phạt $\frac{1}{2}\lambda \|\beta\|_2^2$. Ridge kiểm soát phương sai rất tốt và ổn định trước đa cộng tuyến, nhưng **không thể triệt tiêu bất kỳ hệ số nào về đúng 0** (không có tính năng chọn biến / feature selection). Mô hình vẫn giữ nguyên toàn bộ 54 biến, khó diễn giải.
- **Lasso Regression ($L_1$ Penalty)**: Ép các hệ số về đúng 0 nhờ hình dạng góc nhọn của quả cầu $L_1$ ($\lambda \|\beta\|_1$), tạo ra mô hình thưa (sparse model). Tuy nhiên, khi một nhóm biến có tương quan cao với nhau (như $H_c$ và các tích tương tác của nó), Lasso có xu hướng **chọn bừa 1 biến duy nhất và triệt tiêu tất cả các biến còn lại trong nhóm**, gây mất ổn định và làm suy yếu khả năng khám phá cấu trúc thực tế của dữ liệu.
- **Elastic Net Regression**: Đề xuất bởi Zou & Hastie (2005), kết hợp lồi (convex combination) giữa $L_1$ và $L_2$:
  $$\mathcal{L}_{EN}(\beta) = \frac{1}{2n} \|y - X\beta\|_2^2 + \alpha \left( l_1\_ratio \|\beta\|_1 + \frac{1 - l_1\_ratio}{2} \|\beta\|_2^2 \right)$$
  - Thành phần $L_2$ tạo ra **hiệu ứng nhóm (grouping effect)**: các biến tương quan cao sẽ có xu hướng cùng được giữ lại hoặc cùng được co hệ số với nhau.
  - Thành phần $L_1$ đem lại **tính thưa (sparsity)**: loại bỏ các biến thực sự gây nhiễu về đúng 0.

---

## 2. Chuẩn hóa Thang đo (Standardization) trong Regularization

### Quyết định kỹ thuật
- Bắt buộc áp dụng `StandardScaler(with_mean=True, with_std=True)` cho toàn bộ ma trận thiết kế $X$ (bao gồm cả biến số liên tục, biến nhị phân $0/1$, và các biến tương tác) trước khi đưa vào thuật toán tối ưu `ElasticNet`.
- Thao tác fit của scaler **chỉ được học trên tập train của từng fold tương ứng** và được đóng gói trực tiếp trong `sklearn.pipeline.Pipeline`.

### Căn cứ khoa học
- Hàm phạt $\alpha \sum (l_1 |\beta_j| + \frac{1-l_1}{2}\beta_j^2)$ phạt đồng đều mọi hệ số $\beta_j$ với cùng trọng số $\alpha$.
- Nếu một biến có thang đo lớn (ví dụ `Weekly_GenAI_Hours` $\approx 0 - 40$), hệ số $\beta_j$ của nó tự nhiên sẽ rất nhỏ, do đó bị phạt ít hơn so với một biến dummy nhận giá trị $0/1$. Ngược lại, biến có phương sai nhỏ sẽ có hệ số lớn và bị triệt tiêu oan uổng.
- Chuẩn hóa đưa mọi biến về phân phối có trung bình 0 và độ lệch chuẩn 1, bảo đảm tính công bằng tuyệt đối của hàm phạt.

---

## 3. Quy tắc Tránh Đa cộng tuyến Hoàn hảo trong EN2 (No Raw $H$ with Intercept)

### Quyết định kỹ thuật
- Khi bổ sung $H_c^2$, biến số thô $H$ bị loại bỏ, ma trận thiết kế chỉ giữ lại biến trừ tâm $H_c = H - \overline{H}_{train}$ và bậc hai của nó $H_c^2$.

### Căn cứ toán học
- Theo định nghĩa: $H = H_c + \overline{H}_{train} \cdot \mathbf{1}$.
- Nếu ma trận thiết kế $X$ đã bao gồm cột hằng số 1 (do `fit_intercept=True` tự động cộng vào), việc giữ cả $H$ và $H_c$ sẽ tạo ra một tổ hợp tuyến tính đồng nhất triệt tiêu:
  $$1 \cdot H - 1 \cdot H_c - \overline{H}_{train} \cdot \mathbf{1} = \mathbf{0}$$
- Điều này dẫn đến ma trận bị suy biến (rank deficiency), làm phát sinh vô số nghiệm tương đương và làm sai lệch thuật toán Coordinate Descent của Elastic Net.

---

## 4. Chiến lược Tối ưu hóa Siêu tham số (Coordinate Descent & Grid Search)

### Quyết định kỹ thuật
- Thuật toán giải: **Coordinate Descent** (mặc định và tối ưu nhất của scikit-learn cho Elastic Net).
- Tham số hội tụ: `max_iter=10000`, `tol=1e-4` để loại bỏ hoàn toàn các cảnh báo `ConvergenceWarning` khi $\alpha$ nhỏ hoặc các biến tương quan mạnh.
- Không gian tìm kiếm:
  - $\alpha$: Dải logarithm từ $10^{-4}$ đến $10^1$ (ví dụ: $[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]$).
  - $l_1\_ratio$: $[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]$ để bao phủ từ nghiêng về Ridge đến thuần túy Lasso.
- 5-fold CV chia trên tập train (seed 42) để ước lượng mean CV RMSE và sai số chuẩn.
- Tiêu chí hòa (tie-break): mô hình trong ngưỡng $1.001 \times \text{min\_RMSE}$ được coi là tương đương; ưu tiên mô hình có số hệ số khác 0 ít hơn (độ thưa lớn hơn) để đạt độ bền vững tối đa (Occam's razor).

---

## 5. Phương pháp Luận Trích xuất Chỉ số Đặc thù Elastic Net

### 1. Số lượng Hệ số khác 0 & Tỷ lệ Thưa (Sparsity)
- Trích xuất từ vector trọng số đã fit: $k = \sum_{j=1}^p \mathbb{I}(\hat{\beta}_j \neq 0)$.
- Tỷ lệ thưa: $\text{Sparsity}(\%) = \left(1 - \frac{k}{p}\right) \times 100$.

### 2. Tần suất Chọn Biến qua các Fold (Selection Frequency)
- Huấn luyện mô hình tốt nhất trên từng fold trong 5 fold của CV. Với mỗi đặc trưng $j$, ghi nhận số lần $c_j \in \{0, 1, 2, 3, 4, 5\}$ mà hệ số của nó khác 0.
- Tần suất $f_j = c_j / 5$. Các biến có $f_j = 1.0$ (chọn ở 5/5 folds) được xem là các nhân tố cốt lõi, ổn định và không phụ thuộc vào phương sai mẫu phân chia.

### 3. Kiểm tra Tính Phân cấp (Hierarchical Heredity Check)
- Trong mô hình thống kê cổ điển, nguyên lý phân cấp (hierarchical principle) yêu cầu rằng một số hạng tương tác hai biến $X_1 \times X_2$ chỉ nên xuất hiện nếu cả hai biến thành phần $X_1$ và $X_2$ đều có mặt trong mô hình.
- Elastic Net tiêu chuẩn thực hiện phạt trên từng cột mã hóa độc lập (term-level penalty) mà không có ràng buộc phân cấp cấu trúc. Nghiên cứu này ghi nhận và phân tích các trường hợp vi phạm tính phân cấp như một hiện tượng khách quan và là giới hạn cố hữu của phương pháp.

---

## 6. Giới hạn Nghiên cứu & Khía cạnh Đạo đức ML

1. **Bản chất Dữ liệu Synthetic**: Bộ dữ liệu là nhân tạo. Do đó, các hệ số ước lượng chỉ mô tả mối liên hệ thống kê có điều kiện trong không gian phân phối được sinh ra, tuyệt đối không được ngoại suy thành quan hệ nhân quả trong đời thực.
2. **Kịch bản Hồi cứu (Retrospective Prediction)**: Do thời điểm ghi nhận các biến tâm lý (`Anxiety_Level_During_Exams`, `Perceived_AI_Dependency`) chưa được chứng thực là thu thập trước hay sau kỳ thi, mô hình mang bản chất phân tích điều kiện (What-if analysis) hơn là một hệ thống dự báo triển khai thực tế trước kỳ thi.
3. **Giới hạn Phạt Mức Term vs Group**: Elastic Net có thể chọn một số dummy đại diện của `Primary_Use_Case` và loại bỏ các dummy khác của cùng biến này. Đây là hành vi term-level đặc trưng của Lasso/Elastic Net, khác với Group Lasso vốn loại bỏ hoặc giữ lại nguyên cả biến phân loại.
