# Feature Specification: THÂN — Interaction Linear Regression

**Feature Branch**: Chưa tạo; feature directory `001-interaction-linear-regression`

**Created**: 2026-11-23

**Status**: Draft

**Input**: Notebook tự học bằng tiếng Việt, dự đoán `Post_Semester_GPA` và nghiên cứu association có điều kiện giữa thời gian dùng AI, Prompt Skill và Use Case; kết quả có thể tái lập.

## Clarifications

### Session 2026-11-23

- Chính sách do người dùng cung cấp: dùng lại frozen split hoặc split EDA đã tồn tại sau kiểm tra; nếu thực sự không có split và có raw data thì tạo một lần 70/15/15, seed 42, lưu ID, nhãn và hash nguồn; mọi model dùng cùng assignment.
- Chính sách do người dùng cung cấp: I3 là trọng tâm nghiên cứu, I4 tùy chọn; model test được chọn theo train CV RMSE, dùng tie-break ít terms khi cải thiện rất nhỏ, không chọn theo test.
- Chính sách do người dùng cung cấp: đánh giá I3 no-Anxiety/no-D trên development set; demo chọn chế độ theo features có tại cutoff; categorical reference coding phải cảnh báo category chưa thấy.
- Chính sách do người dùng cung cấp: ưu tiên ipywidgets nếu hỗ trợ, luôn có `predict_student(raw_dict, model)`; thiếu data báo lỗi kèm đường dẫn, không khẳng định notebook đã chạy.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Hiểu bài toán và interaction (Priority: P1)

Sinh viên mới học Linear Regression đọc notebook từ đầu đến cuối để hiểu dữ liệu, mục tiêu, leakage, split và ý nghĩa mô hình interaction.

**Why this priority**: Hiểu assumptions và tránh hiểu sai kết quả là mục đích học tập cốt lõi.

**Independent Test**: Đọc phần mở đầu và giải lại ví dụ minh họa bằng tay, không cần chạy huấn luyện.

**Acceptance Scenarios**:

1. **Given** notebook được mở, **When** người học đọc phần giới thiệu, **Then** thấy động cơ, 12 inputs, một target, các biến bị loại, vai trò train/validation/test và sự khác nhau giữa simple linear regression và interaction.
2. **Given** ví dụ minh họa, **When** người học đối chiếu định nghĩa Hc, Pc, Sc, Ui, Kj và công thức I0–I4, **Then** có thể kiểm tra phép tính tay và giải thích reference group cùng conditional slope.

### User Story 2 — Chạy dữ liệu và EDA (Priority: P1)

Sinh viên chạy từ kernel sạch, kiểm tra dữ liệu đã chia sẵn và quan sát EDA trước khi huấn luyện.

**Why this priority**: Dữ liệu hợp lệ và bối cảnh đo lường là nền tảng của mọi kết quả.

**Independent Test**: Cung cấp các phân vùng hợp lệ, chạy đến hết EDA và kiểm tra báo cáo đầu vào.

**Acceptance Scenarios**:

1. **Given** dữ liệu chia sẵn hợp lệ, **When** chạy từ đầu, **Then** notebook kiểm tra schema, target, phân vùng và hiển thị EDA trước training; nêu nguồn, bản chất synthetic theo bằng chứng và availability của features.
2. **Given** dữ liệu hoặc cột bắt buộc bị thiếu, **When** người học chạy, **Then** nhận thông báo có tên mục thiếu và cách khắc phục; không có metrics hoặc plots giả thay thế.

### User Story 3 — Huấn luyện I0–I4 (Priority: P1)

Sinh viên đối chiếu I0–I3 là các mô hình OLS lồng nhau để hiểu từng nhóm interaction; I3 là trọng tâm và I4 là mở rộng tùy chọn.

**Why this priority**: Đây là nội dung mô hình cốt lõi của phần THÂN.

**Independent Test**: Với dữ liệu đủ levels, kiểm tra danh sách terms của I0–I3 và I4 nếu được bật.

**Acceptance Scenarios**:

1. **Given** đủ category levels, **When** xem model specifications, **Then** I0–I3 có 39, 41, 45, 53 predictor terms, không tính intercept; I4 nếu bật có 55; tất cả đúng thứ tự lồng nhau và đủ main effects.
2. **Given** training hoặc một CV fold, **When** chuẩn bị các mô hình, **Then** reference coding và centering chỉ học từ training tương ứng; I0–I4 không có quadratic, spline, hinge hoặc interaction bậc ba.

### User Story 4 — Chọn mô hình trước test (Priority: P1)

Sinh viên dùng CV trên train và validation để chọn mô hình theo chính sách đã công bố trước khi xem test.

**Why this priority**: So sánh tái lập và bảo vệ holdout là điều kiện đánh giá đáng tin cậy.

**Independent Test**: Kiểm tra bảng CV/validation, lựa chọn được ghi lại và lịch sử truy cập test.

**Acceptance Scenarios**:

1. **Given** cùng split và folds, **When** đánh giá I0–I3 và I4 nếu bật, **Then** hiển thị MAE, MSE, RMSE, R² cho từng fold, CV mean ± standard deviation và validation cho từng mô hình.
2. **Given** policy chọn theo mean 5-fold CV RMSE đã công bố, **When** model nằm trong 0,1% RMSE nhỏ nhất, **Then** chọn model ít terms nhất; ghi quyết định và khóa trước test, không dùng test để chọn. Nếu RMSE nhỏ nhất bằng 0 thì chỉ giá trị đúng 0 là tương đương.

### User Story 5 — Đánh giá final holdout (Priority: P1)

Sinh viên nhận kết quả test cuối cùng và hiểu sai số dự đoán thực tế.

**Why this priority**: Final holdout cung cấp đánh giá cuối không dùng để tuning.

**Independent Test**: Sau model lock, đối chiếu metrics với predictions đã lưu và xem các biểu đồ bắt buộc.

**Acceptance Scenarios**:

1. **Given** mô hình và protocol đã khóa, **When** chạy final evaluation, **Then** chỉ đánh giá final test một lần cho run đó và báo cáo MAE, MSE, RMSE, R², counts cùng tỷ lệ dự đoán <0 và >4; không clip trước đánh giá.
2. **Given** kết quả test thật, **When** xem phần đánh giá, **Then** thấy actual-vs-prediction, residual-vs-fitted, histogram residual và nhận xét cụ thể dựa trên số liệu; test không quay lại điều chỉnh lựa chọn.

### User Story 6 — Diễn giải interaction (Priority: P1)

Sinh viên khám phá conditional slopes và contrasts theo Use Case × Prompt Skill khi giữ các input còn lại cố định.

**Why this priority**: Giải thích interaction là mục tiêu nghiên cứu, không chỉ dự đoán.

**Independent Test**: Chọn một profile và hai nhóm, tính lại slope và contrast từ các hệ số được báo cáo.

**Acceptance Scenarios**:

1. **Given** profile và reference levels được nêu, **When** so sánh nhóm Use Case × Prompt Skill, **Then** thấy conditional slopes, joint contrasts đầy đủ main effects và interaction, predicted GPA curves hoặc heatmaps cùng phép tính có thể kiểm tra.
2. **Given** biểu đồ what-if và kết quả interaction, **When** đọc diễn giải, **Then** thấy rõ association/prediction không chứng minh tác động nhân quả.

### User Story 7 — Kiểm tra độ ổn định và giới hạn (Priority: P2)

Sinh viên xem sai số theo nhóm, ablation, độ đa cộng tuyến và các ca thất bại để hiểu giới hạn kết quả.

**Why this priority**: Các phân tích này bổ sung chiều sâu sau luồng huấn luyện chính.

**Independent Test**: Đối chiếu subgroup counts/RMSE, từng nhóm ablation và diagnostics với nguồn kết quả.

**Acceptance Scenarios**:

1. **Given** kết quả I0–I3, I4 nếu bật và biến thể sensitivity chạy được, **When** xem bảng phân tích, **Then** có subgroup RMSE/counts theo Use Case và Prompt Skill, group ablation I0→I3 (thêm I4 nếu bật) và I3 full/no Anxiety/no D trên cùng development split/CV.
2. **Given** diagnostics và failure cases, **When** đọc giới hạn, **Then** thấy VIF/condition number và định nghĩa ma trận, cảnh báo nhóm nhỏ, dữ liệu synthetic, uncertainty và không có khẳng định statistical significance chưa kiểm chứng.

### User Story 8 — Dự đoán và what-if (Priority: P1)

Người dùng nhập 12 raw inputs hoặc tập con hợp lệ cho mô hình no-D để nhận GPA dự đoán và khám phá thay đổi AI Hours.

**Why this priority**: Giao diện minh họa là deliverable Lab và kết nối bài học với ứng dụng.

**Independent Test**: Nhập một hàng hợp lệ, thay đổi AI Hours và đối chiếu kết quả với mô hình đã lưu.

**Acceptance Scenarios**:

1. **Given** mô hình tương ứng đã huấn luyện, **When** nhập 12 inputs hoặc tập con đúng schema no-D, **Then** hiển thị GPA dự đoán, model/scenario và đồ thị what-if; không bắt nhập engineered features.
2. **Given** giá trị ngoài miền hoặc category chưa biết, **When** dự đoán, **Then** hiển thị cảnh báo; category chưa thấy được mã hóa thành vector toàn 0 và giải thích rằng không phân biệt được reference trong encoding; khi widget không hoạt động, dùng `predict_student(raw_dict, model)`.

### User Story 9 — Bàn giao và tái lập (Priority: P1)

Người đánh giá theo Quickstart để chạy và kiểm tra toàn bộ bài trên Windows hoặc Kaggle/Colab bằng CPU.

**Why this priority**: Bài chỉ có giá trị khi người khác có thể mở, chạy và kiểm chứng.

**Independent Test**: Làm theo README trong môi trường sạch với split hợp lệ và kiểm tra artifacts cùng kết quả test.

**Acceptance Scenarios**:

1. **Given** môi trường và dữ liệu đã cấu hình, **When** thực hiện Restart/Run All theo Quickstart, **Then** thu notebook executed, bảng, PNG, fitted model pipeline, provenance và kết quả tests; không cần web server hoặc database.
2. **Given** split thật không sẵn có, **When** chạy kiểm tra, **Then** thông báo actionable và ghi `real training: not executed`; fixture chỉ dùng kiểm thử và không được báo là kết quả benchmark.

### Edge Cases

- Split đã tồn tại nhưng thiếu một partition, schema/target thiếu, hash không khớp hoặc ID/samples chồng lấn: dừng, chỉ rõ lỗi; không tái chia hay chuyển dòng để che lỗi. Chỉ tạo split khi hoàn toàn chưa có partition hoặc assignment nào.
- Thiếu ID: kiểm tra fingerprint/trùng hoàn toàn khả dụng và nêu giới hạn xác minh leakage theo nguồn.
- Thiếu category level trong train hoặc fold: báo levels, số term thực tế và ảnh hưởng; không nhìn validation/test hoặc thêm cột giả để ép counts.
- Category chưa thấy khi inference: cảnh báo và mã hóa all-zero theo reference coding; báo hạn chế đồng nhất với reference. Giá trị thiếu/sai kiểu bắt buộc: báo lỗi; giá trị ngoài miền hợp lệ: cảnh báo, không tự clip.
- Target không hợp lệ/thiếu, variance bằng 0, subgroup quá nhỏ, VIF không xác định hoặc ma trận suy biến: báo lý do, số mẫu và giới hạn; không khẳng định metric có ý nghĩa.
- Widget không được hỗ trợ, không có GPU/network, hoặc dữ liệu không sẵn: dùng fallback phù hợp; không giả lập kết quả học thật.
- Test đã bị dùng để tuning: đánh dấu kết quả nhiễm và chặn tuyên bố holdout độc lập cho đến khi có protocol mới được phê duyệt.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Notebook MUST trình bày bằng tiếng Việt, có LaTeX, phần mục đích/input/output/cách đọc kết quả, code cells vừa phải có comment và thứ tự học từ bài toán, schema, EDA, ví dụ tay đến huấn luyện, diễn giải, đánh giá và demo.
- **FR-002**: Input đầy đủ MUST gồm `Pre_Semester_GPA`, `Traditional_Study_Hours`, `Year_of_Study`, `Major_Category`, `Weekly_GenAI_Hours`, `Primary_Use_Case`, `Prompt_Engineering_Skill`, `Tool_Diversity`, `Paid_Subscription`, `Institutional_Policy`, `Perceived_AI_Dependency`, `Anxiety_Level_During_Exams`; target duy nhất là `Post_Semester_GPA`.
- **FR-003**: `Student_ID`, target, `GPA_Change`, `Skill_Retention_Score`, `Burnout_Risk_Level` và các biến dẫn xuất từ target MUST bị loại khỏi predictors; feature chưa sẵn có tại prediction cutoff MUST không dùng trong kịch bản dự báo tương ứng. Mô hình có nhóm D khi timing chưa rõ MUST gắn nhãn conditional/retrospective.
- **FR-004**: Dùng nguyên vẹn `data/splits/frozen_split_ids.csv` hoặc assignment split EDA/nhóm nếu có, sau kiểm tra source hash, ID duy nhất, coverage, labels, tỷ lệ, overlap, group integrity và mapping sang phân vùng; hỗ trợ ba tệp hoặc một tệp có nhãn split. Nếu hoàn toàn không có assignment hay partition và có raw data hợp lệ, tạo đúng một split 70/15/15 với random_state=42, lưu `Student_ID`, nhãn train/validation/test, source hash, rule và counts; không chia lại nếu split hiện có lỗi. Nếu nhiều split bất đồng, chặn so sánh cho đến khi nhóm xác nhận split chung; không chọn âm thầm. I0–I3, I4 nếu bật và mọi ablation MUST dùng cùng frozen assignment; lưu manifest và split ID.
- **FR-005**: EDA MUST trước training, dùng train cho phân tích mô hình, nêu nguồn, synthetic status theo bằng chứng, feature availability và bối cảnh; không phân tích test để thiết kế mô hình.
- **FR-006**: OLS I0 MUST có main effects; I1 thêm Hc×K1..K2; I2 thêm Hc×U1..U4; I3 thêm Ui×Kj với i=1..4, j=1..2; I4 tùy chọn thêm Hc×Pc và Hc×Sc. Khi đủ levels số predictor terms tương ứng là 39/41/45/53 và 55 cho I4 nếu bật, không tính intercept. I3 là primary cho nghiên cứu và diễn giải.
- **FR-007**: H, P, S lần lượt là Weekly_GenAI_Hours, Pre_Semester_GPA, Traditional_Study_Hours; Hc/Pc/Sc trừ mean training tương ứng. Ui là bốn dummy Use Case, Kj là hai dummy Prompt Skill. Year/Major/Use/Prompt/Policy/Dependency/Anxiety dùng reference-coded categorical bỏ một reference; Dependency và Anxiety không được coi là continuous. Ghi reference levels và mapping Paid_Subscription. Category chưa thấy dùng all-zero và cảnh báo vì không phân biệt được reference; verify levels và counts chỉ từ train/fold hoặc schema cố định trước, không dò validation/test. Giữ main effects, không thêm quadratic/spline/hinge/ba chiều vào benchmark.
- **FR-008**: Mọi fitted preprocessing, category vocabulary, imputation và centering MUST học chỉ từ training partition trong mỗi fold; giữ cùng feature definitions, prediction scenario và split trong so sánh. Không dùng test mean/statistics để fit hay điều chỉnh.
- **FR-009**: Dùng 5-fold CV trên train; ưu tiên giữ group cùng fold khi có group ID, nếu không dùng folds shuffle với seed 42. Ghi folds, metrics từng fold, mean ± sd cho MAE/MSE/RMSE/R² và bảng validation của I0–I3, I4 nếu bật.
- **FR-010**: Chọn model chính theo mean CV RMSE thấp nhất trong tập candidates đã khai báo trước: I0–I3 và I4 chỉ khi bật trước CV. Mặc định coi model có RMSE ≤ 1,001 lần RMSE nhỏ nhất là tương đương và chọn ít predictor terms nhất; nếu vẫn hòa chọn chỉ số I nhỏ hơn. Nếu min RMSE = 0, chỉ model RMSE = 0 tương đương. Validation chỉ mô tả/xác nhận, không thay lựa chọn CV; báo I3 riêng trên development set dù không được chọn. Ghi model/scenario và khóa trước test. Model cuối và preprocessing chỉ fit trên train; validation và test giữ riêng. Test chỉ final evaluation một lần mỗi run, không tuning.
- **FR-011**: Báo metrics final test MAE/MSE/RMSE/R², counts, tỷ lệ <0/>4 và các residual/actual-predicted plots; metrics MUST tính lại được từ prediction và nhãn. Không yêu cầu giá trị hiệu năng định trước.
- **FR-012**: Cho thấy coefficients, intercept, conditional slopes I3 và I4 nếu bật, contrast Use×Prompt tại profile cố định, predicted curves/heatmaps, nhóm reference, các giả định và cách tính; giải thích what-if không phải can thiệp nhân quả.
- **FR-013**: Có I0→I3 ablation (I4 nếu bật), I3 full/no Anxiety/no D trên CV train và validation, subgroup RMSE/counts theo Use Case và Prompt Skill, residual theo fitted/AI Hours/Pre GPA, VIF/condition number kèm định nghĩa ma trận, failure cases và giới hạn/uncertainty. Cảnh báo subgroup có dưới 30 mẫu; không dùng test để chọn biến thể.
- **FR-014**: Giao diện notebook ưu tiên ipywidgets nếu hỗ trợ, luôn cung cấp `predict_student(raw_dict, model)` nhận raw inputs, trả dự đoán, model/scenario và cảnh báo. Cung cấp chế độ full, no-Anxiety và no-D phù hợp dữ liệu sẵn có tại cutoff, mỗi chế độ dùng model tương ứng đã fit; trước kỳ thi không cho chọn model cần exam anxiety chưa có. Hiển thị what-if AI Hours; Streamlit là tùy chọn, không là dependency bắt buộc.
- **FR-015**: Tạo tối thiểu 10 đồ thị khác mục đích phân tích, lấy từ dữ liệu/run thật; mỗi đồ thị có title, labels, chú thích, tập dữ liệu và 1–3 câu nhận xét dựa trên số liệu. Chỉ ghi `not executed` khi chưa có output thật.
- **FR-016**: Lưu notebook executed, bảng, hình PNG, fitted model pipeline và metadata về input hashes, split, folds, seed, versions, reference levels, model, metrics, code revision nếu có và run; README có Quickstart, schema, lệnh chạy/test, limitations và references.
- **FR-017**: Tests MUST kiểm tra schema/exclusions, split identity/disjointness và tái sử dụng; trường hợp không có split kiểm tra tạo một lần rồi tái sử dụng, trường hợp split lỗi kiểm tra fail không tái chia; reference coding/unseen warning, fold-only centering, hierarchy/counts theo train, tie-break, no test selection, modes theo timing, finite prediction, fallback, model reload và notebook format/execution smoke. Fixture chỉ dùng test, không làm benchmark.
- **FR-018**: So sánh với Ridge, Elastic Net, Polynomial, Piecewise chỉ khi có outputs tương thích về target, split, metric và prediction scenario; thiếu outputs thì ghi chưa có, không tự tạo kết quả hoặc sửa phần của thành viên khác.
- **FR-019**: Mỗi figure và bảng liên quan MUST ghi run ID, split/fold hoặc partition, model và metric; figure so sánh MUST dùng cùng values với bảng nguồn. Trước xuất, đối chiếu số điểm, nhãn và giá trị figure với bảng cùng run theo tolerance công bố; sai lệch MUST báo lỗi và không tuyên bố đạt.

### Nonfunctional Requirements

- **NFR-001 — Tái lập**: Cùng dữ liệu, split, configuration và environment đã ghi phải cho phép tái tính tất cả metrics; so sánh floating-point theo tolerance được công bố trước.
- **NFR-002 — Khả dụng**: Restart/Run All từ kernel sạch chạy trên Windows hoặc Kaggle/Colab bằng CPU khi dữ liệu và environment hợp lệ; không cần web server, database hoặc network mặc định để lấy data.
- **NFR-003 — Minh bạch**: Mọi kết quả, bảng và figures có nguồn/run provenance; ghi `not executed` cùng lý do nếu chưa chạy, không bịa số; người nộp review và hiểu nội dung AI hỗ trợ.
- **NFR-004 — Học tập**: Giải thích dễ hiểu cho sinh viên biết Linear Regression cơ bản; không khẳng định nhân quả hoặc significance khi chưa chứng minh và không ẩn uncertainty/limitations.

### Key Entities *(include if feature involves data)*

- **Student record**: 12 raw features, target liên tục khi huấn luyện, ID chỉ phục vụ kiểm tra; availability phụ thuộc prediction scenario.
- **Frozen partition**: Tập train, validation, test có assignment ổn định, fingerprint, counts và checks; dùng chung giữa các thành viên.
- **Model specification**: I0–I4 hoặc biến thể sensitivity; term set, reference levels, centering và scenario xác định.
- **Experiment run**: Split/folds, phiên bản, configuration, metrics, predictions, figures, model đã fit và provenance.
- **Prediction profile**: Một hàng raw inputs hợp lệ cùng model/scenario, cảnh báo và dự đoán/what-if.

### Out of Scope

- Thay đổi hoặc huấn luyện thay các mô hình của thành viên khác; tự tạo kết quả so sánh khi chưa có outputs.
- Deep learning, suy luận causal effect, hoặc dùng test để chọn feature/model và điều chỉnh protocol.
- Bắt buộc web server hay database; giả lập metrics, plots hoặc kết luận khi thiếu dữ liệu thật.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Với CSV đã cấp 50.000 bản ghi và partition hợp lệ, xác minh đúng 35.000/7.500/7.500 train/validation/test, không chồng lấn theo khóa kiểm tra và không tái chia.
- **SC-002**: Khi train đủ levels, I0–I3 có đúng 39/41/45/53 predictor terms không tính intercept; I4 nếu bật có 55; mỗi interaction giữ main effects và không có quadratic/spline.
- **SC-003**: Có ít nhất 10 đồ thị khác mục đích, mỗi hình có title, labels, tập dữ liệu và 1–3 câu diễn giải dựa trên số liệu run thật.
- **SC-004**: Tất cả MAE/MSE/RMSE/R² đã báo cáo tính lại được từ dự đoán và nhãn tương ứng theo tolerance công bố; bảng CV có 5 folds và mean ± sd cho từng metric; test chỉ sau model lock.
- **SC-005**: Một profile hợp lệ cho đầy đủ 12 inputs và một profile no-D đều dự đoán được; thay AI Hours tạo được what-if và fallback hoạt động khi widget không hỗ trợ.
- **SC-006**: Các kiểm tra leakage, split identity/disjointness, fold-only fitting, encoding, unseen warning, model selection và term hierarchy/counts đều passed; có ablation I0→I3 và sensitivity full/no Anxiety/no D trên development set khi dữ liệu cho phép.
- **SC-007**: Người đánh giá theo README có lệnh cài/chạy/test, có thể chạy sạch và thu notebook output thật, bảng, PNG, fitted model, references và provenance; nếu thiếu dữ liệu thật thì kết quả thực được đánh dấu `not executed`.
- **SC-008**: Người học có thể đối chiếu một ví dụ tính tay, một conditional slope và một joint contrast với công thức/hệ số; mỗi nhận xét dự đoán nêu rõ không hàm ý nhân quả.

## Assumptions

- Split hiện có là ứng viên dùng lại, không tạo split mới trong workspace hiện tại. Kiểm tra ngày 2026-11-23: raw SHA-256, assignment SHA-256, hashes ba partition và bản sao khớp metadata; 50.000 IDs duy nhất, assignment khớp từng partition, counts 35.000/7.500/7.500, giao ID bằng 0, không có bản ghi nội dung hoàn toàn trùng giữa partition. Metadata xác nhận split này khác EDA gốc; assignment EDA gốc chưa thấy, không mặc định nhóm đã phê duyệt hay hai split tương đương.
- Theo constitution 2.0.0 đề xuất, 39/41/45/53/55 đếm predictors không tính intercept; schema đủ levels và các công thức I0–I4 đã nêu là hợp đồng feature hiện tại. Mỗi nhóm reference xác định từ train hoặc schema cố định trước đánh giá.
- Validation chỉ xác nhận; tie threshold 0,1% và unknown-category all-zero kèm cảnh báo là mặc định của đặc tả, được công bố trước CV. Mặc định không refit train+validation nhằm giữ cùng fitted model cho validation/test.
- Constitution 2.0.0 đề xuất đã đồng bộ create-once và I4 tùy chọn với chỉ thị người dùng; model cuối fit train-only theo chỉ thị implementation mới nhất. Ngày và quyết định phê duyệt của nhóm chưa được xác nhận.
- PDF Lab 01 được đọc qua [bản tài liệu trên Drive](https://drive.google.com/file/d/192i0AvnmsLPdTSn_c1bnl87_2khKc1b7/view); yêu cầu nộp nhóm như báo cáo PDF, video và ZIP thuộc bàn giao chung, không bị coi là đã hoàn thành bởi feature notebook này.
- `REFERENCE_REQUIREMENTS.md` chưa tìm thấy; tìm kiếm Drive chưa xác định được Docs/tab THÂN tương ứng. Các yêu cầu chi tiết hiện dựa trên mô tả người dùng và constitution trong repo; khi có nguồn, đối chiếu và ghi nhận khác biệt theo governance, không chặn việc soạn spec.
- Nếu dataset/source metadata chưa chứng minh synthetic hoặc timing, nêu `unverified` và không phát biểu như sự thật; notebook thực tế cần dẫn chứng trước khi khẳng định.

### Open Issues for `/speckit.clarify`

1. Xác nhận vị trí `REFERENCE_REQUIREMENTS.md` và Docs/tab THÂN để đối chiếu với hợp đồng hiện tại.
2. Xác nhận split hiện có được năm thành viên dùng chung; nếu tìm được assignment EDA gốc và khác, cần quyết định có ghi nhận trước khi chạy so sánh.
3. Ghi nhận phê duyệt constitution 2.0.0 của nhóm khi thực sự có; không giả định đã được thông qua.
