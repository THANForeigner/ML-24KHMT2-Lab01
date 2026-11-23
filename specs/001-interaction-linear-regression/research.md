# Research: Interaction Linear Regression

Ngày: 2026-11-23. Phạm vi là quyết định thiết kế; không có benchmark mới trong bước này.

## Split, source và governance

**Decision:** Ưu tiên assignment nhóm đã xác nhận, kiểm tra raw SHA-256, membership/coverage, ID, duplicates, group overlap, hashes và partitions trước khi dùng; toàn bộ I0–I4/ablation cùng split. Assignment hiện có `data/assignments.csv` khớp ba partitions, raw hash và 35k/7,5k/7,5k; metadata nói khác split EDA gốc nên không ngụ ý nhóm đã duyệt. Nếu hai assignment cùng sẵn có mà khác, dừng và ghi mismatch.

**Rationale:** Kết quả chỉ so sánh công bằng trên cùng population/split; test phải là holdout.

**Alternatives:** Tạo split mới mỗi run bị loại; create-once khi hoàn toàn chưa có split là chỉ thị mới, nhưng xung đột constitution 1.1.0 nên chỉ triển khai sau khi đồng bộ governance. Không resplit khi split lỗi.

## Reference encoding và transform

**Decision:** Median train-only, impute rồi tính means H/P/S trên training partition, categorical mode train-only (tie deterministic), all-missing fail; Dependency/Anxiety treated categorical kể cả dtype số. `OneHotEncoder(drop='first', handle_unknown='ignore')`; explicit warning theo fitted `categories_` cho từng input; unknown all-zero trùng baseline, ảnh hưởng cả interactions. Dùng `categories_`/`drop_idx_` và mapping feature metadata để tìm Ui/Kj, không suy đoán tách chuỗi tên có thể chứa dấu gạch dưới.

**Rationale:** Fold-specific state tránh leakage; category levels được học tại từng fold. [OneHotEncoder documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) xác nhận unknown+ignore được mã hóa all-zero.

**Alternatives:** `handle_unknown='error'` không đáp ứng demo; explicit unseen category indicator thay đổi contract; lookup levels từ test là leakage. Không ép đủ 39–55 khi fold thiếu mức; test fixture đủ levels kiểm count, real run báo sai khác.

## OLS, rank và diễn giải

**Decision:** Pipeline cloneable, dense OLS `LinearRegression(fit_intercept=True)`, không regularize/scale mặc định. Log SVD/rank và singular values cho design matrix kèm intercept policy, tolerance, condition number; VIF train-only, đánh dấu infinite/undefined thay vì thay số tùy tiện. [LinearRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) mô tả OLS, rank và singular values cho dense input.

**Rationale:** Không tự nghịch đảo `XᵀX`; scaling mặc định không cần cho OLS và giữ units dễ dạy. Nếu bật scaling, fit trong fold; quy đổi cả intercept và interaction term về raw design, kiểm slopes/contrast bằng dự đoán profile nguyên gốc. Với `fit_intercept=True`, `rank_` của estimator là rank trên design được center nội bộ; phân biệt với rank của augmented matrix có intercept.

**Alternatives:** Ridge/Elastic Net chỉ là comparison ngoài nhánh, không thay OLS; loại biến do test để sửa singularity là leakage.

## Evaluation, lock và final test

**Decision:** Common 5-fold training, group-aware nếu có group; còn lại shuffle seed 42. Chọn dựa mean CV RMSE, chấp nhận ≤1,001×best rồi ít terms nhất, tie theo chỉ số I; nếu best=0 chỉ exact-zero. Validation sanity/descriptive, I3 báo riêng. Theo chỉ thị implementation mới nhất, model cuối fit trên train và đánh giá test một lần sau lock; không fit trên validation. I3 no-D/no-anxiety development only, trừ khi một biến thể được khai báo là model final trước lock.

**Rationale:** Giữ validation và test tách khỏi dữ liệu fit theo chỉ thị mới nhất; lưu provenance của train-only model và model lock.

**Alternatives:** Chọn theo validation hoặc test bị loại; report final test của mọi model không cần thiết và không cho dùng để chọn. Replaying cùng lock chỉ nhằm tái lập, không cho dùng để thử nhiều candidate.

## Demo, portability và bảo mật artifact

**Decision:** Modes full/no-anxiety/no-D, mỗi mode gắn fitted artifact/cutoff; widget optional, fallback `predict_student(raw_dict, model)`; unknown/domain warnings; no clipping. Notebook tìm root dựa marker hoặc env var và input path configurable, chạy offline sau cài dependency. Joblib chỉ đọc artifact do code tin cậy tạo.

**Rationale:** Dữ liệu D/anxiety chưa chắc sẵn tại prediction time; không thể dùng một model đầy đủ bằng cách tự điền inputs bị thiếu. Code pickle/joblib có thể thực thi khi load; không coi model từ nguồn lạ là an toàn.

**Alternatives:** Bắt buộc widget/server bị loại; dùng test để lấy category vocabulary bị loại. Các nhóm còn lại có timing chưa rõ đều phải gắn nhãn retrospective nếu chưa xác minh, kể cả no-D.

## Tài liệu kỹ thuật

- [scikit-learn: common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- [scikit-learn: pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html).
- [nbclient execution](https://nbclient.readthedocs.io/en/latest/client.html).

Các issue ngoài kỹ thuật vẫn mở: xác nhận split chung; đồng bộ constitution/spec theo chỉ thị mới; đối chiếu `REFERENCE_REQUIREMENTS.md` và tab THÂN khi có. Không giả định các điểm này đã được phê duyệt.
