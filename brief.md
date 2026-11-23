# Project Brief — AI Impact on Students: Exploratory Data Analysis

> **Deliverable:** Một Jupyter Notebook EDA hoàn chỉnh, có thể chạy lại từ đầu đến cuối và tự sinh các bảng thống kê, biểu đồ, nhận xét có bằng chứng và khuyến nghị phục vụ Linear Regression.
>
> **Phạm vi:** Lab 01 – Linear Regression, Introduction to Machine Learning.

## 1. Bối cảnh và mục tiêu

Dataset [AI Impact on Students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students) mô tả các thuộc tính học tập, hành vi sử dụng Generative AI và kết quả học tập của sinh viên. Dự án hướng tới dự đoán GPA cuối học kỳ bằng các mô hình hồi quy tuyến tính và đánh giá giá trị dự đoán bổ sung của nhóm thuộc tính AI so với nhóm thuộc tính học tập truyền thống.

EDA là giai đoạn chuẩn bị dữ liệu và thiết kế mô hình. Notebook phải giúp nhóm trả lời: dữ liệu có đáng tin cậy cho mục đích dự đoán hay không, đặc trưng nào có ích, đặc trưng nào có nguy cơ rò rỉ, quan hệ nào cần biến đổi, và nên xây dựng những mô hình hồi quy nào.

### 1.1. Kết quả cần đạt

- Khảo sát schema, chất lượng và phân phối của toàn bộ thuộc tính thực tế.
- Phân tích biến mục tiêu, mối quan hệ đơn biến, hai biến và đa biến.
- Kiểm tra tuyến tính, đa cộng tuyến, các giả định và phần dư của mô hình hồi quy thăm dò.
- Xây dựng cơ sở có bằng chứng cho preprocessing, feature engineering, feature selection và thiết kế nhiều mô hình Linear Regression.
- Tự động sinh và lưu bảng, biểu đồ, báo cáo EDA và thông tin chia tập để các thành viên sử dụng thống nhất.
- Bảo đảm khả năng tái lập, không rò rỉ dữ liệu, không sửa dữ liệu gốc và không tạo kết quả thống kê giả.

### 1.2. Phạm vi

**In scope:** EDA, structural audit, data cleaning tối thiểu có quy tắc, split, phân tích đặc trưng, hồi quy thăm dò, diagnostics, khuyến nghị mô hình, export artifacts, kiểm tra khả năng chạy notebook.

**Out of scope:** huấn luyện/tuning toàn bộ mô hình cuối cùng, đánh giá final test, triển khai web/API, deep learning, classification, thu thập dữ liệu mới, suy luận nhân quả. Không xây dựng các notebook training hoặc thay đổi file dự án không liên quan trong feature này.

### 1.3. Nguồn và mức độ xác minh

| Nguồn | Nội dung sử dụng |
|---|---|
| Kaggle dataset URL ở trên | Nguồn dữ liệu cần tải hoặc gắn vào runtime; xác minh tên file, phiên bản, schema, license, provenance |
| `ml-data.pdf` (trang 1–2) | Hướng dẫn báo cáo nguồn, chất lượng, làm sạch, thống kê mô tả, phân phối nhãn, tương quan, đa cộng tuyến và ngoại lệ |
| `Lab 01 - Linear Regression.pdf` (trang 1–3) | Bài toán biến mục tiêu liên tục, Linear Regression, giải thích lựa chọn biến, so sánh mô hình và metrics, số mô hình tối thiểu bằng số thành viên |

Schema trong brief là **schema dự kiến**, không phải kết quả đọc CSV trong lần soạn brief. Quy mô tham khảo khoảng 50.000 dòng, 16 cột; nguồn gốc synthetic và giấy phép cần được xác minh từ metadata chính thức. Không hardcode số liệu, tương quan, kết quả test, miền giá trị chưa xác minh hoặc kết luận từ các notebook công khai. Nếu không xác minh được provenance/đơn vị/thời điểm thu thập, ghi `unknown` hoặc `unverified` và nêu hệ quả.

## 2. Định nghĩa bài toán

- **Primary target:** `Post_Semester_GPA` (continuous regression).
- **Secondary analytical target:** `GPA_Change = Post_Semester_GPA - Pre_Semester_GPA`. Chỉ dùng phân tích hoặc làm target cho bài toán hồi quy riêng; tuyệt đối không làm predictor cho Post GPA.
- **Prediction scenario:** dự đoán GPA cuối kỳ tại một thời điểm trước khi kết quả cuối kỳ được biết. Mặc định bảo thủ: chỉ đưa vào mô hình dự đoán chính thuộc tính được xác minh có sẵn lúc dự đoán. Nếu metadata không xác định thời điểm thu thập, nhóm biến AI/hành vi chỉ được dùng cho phân tích liên hệ hoặc mô hình thăm dò có gắn nhãn `conditional / retrospective`, không được tuyên bố là mô hình dự báo triển khai được.
- Các mô hình chính phải dùng cùng target, sample population, split và metric để so sánh công bằng.
- Không suy ra AI gây tăng/giảm GPA từ tương quan, chênh lệch trung bình hoặc hệ số hồi quy.

### 2.1. Expected Data Dictionary

| Column | Ý nghĩa dự kiến | Semantic type | Vai trò dự kiến |
|---|---|---|---|
| `Student_ID` | Định danh sinh viên | Identifier | Group/split audit; không làm predictor |
| `Major_Category` | Nhóm ngành | Nominal | Candidate predictor |
| `Year_of_Study` | Năm học | Ordinal | Candidate predictor |
| `Pre_Semester_GPA` | GPA trước học kỳ | Continuous | Baseline predictor |
| `Weekly_GenAI_Hours` | Giờ sử dụng GenAI mỗi tuần | Continuous | Conditional predictor |
| `Primary_Use_Case` | Mục đích dùng AI chính | Nominal | Conditional predictor |
| `Prompt_Engineering_Skill` | Mức kỹ năng prompt | Ordinal | Conditional predictor |
| `Tool_Diversity` | Số công cụ AI sử dụng | Discrete | Conditional predictor |
| `Paid_Subscription` | Đăng ký AI trả phí | Binary | Conditional predictor |
| `Traditional_Study_Hours` | Thời gian học truyền thống | Continuous | Conditional predictor; xác minh đơn vị |
| `Perceived_AI_Dependency` | Mức phụ thuộc AI tự đánh giá | Ordinal | Conditional predictor |
| `Institutional_Policy` | Chính sách AI của trường | Nominal | Candidate predictor |
| `Anxiety_Level_During_Exams` | Mức lo lắng khi thi | Ordinal | Conditional/high temporal risk |
| `Post_Semester_GPA` | GPA sau học kỳ | Continuous | Primary target |
| `Skill_Retention_Score` | Điểm duy trì kiến thức/kỹ năng | Continuous | Auxiliary outcome; mặc định loại khỏi predictors |
| `Burnout_Risk_Level` | Mức nguy cơ kiệt sức | Ordinal | Auxiliary outcome; mặc định loại khỏi predictors |

Data Dictionary sinh từ CSV phải có tên cột thực tế, mô tả, dtype quan sát, semantic type, đơn vị nếu biết, miền hợp lệ nếu xác minh, min/max hoặc categories quan sát, missing count, unique count, vai trò, thời điểm khả dụng và nguồn xác minh. Không xem ID là biến số có ý nghĩa chỉ vì dtype là integer.

### 2.2. Research Questions

| ID | Câu hỏi | Section chính |
|---|---|---|
| RQ01 | Dữ liệu đầy đủ, nhất quán và phù hợp cho regression đến mức nào? | 2–4 |
| RQ02 | Pre GPA dự đoán Post GPA tốt đến mức nào? | 6, 7, 9 |
| RQ03 | AI Hours liên hệ như thế nào với Post GPA và GPA Change? | 7, 8 |
| RQ04 | Traditional Study Hours liên hệ như thế nào với kết quả học tập? | 7, 8 |
| RQ05 | Prompt Skill, Primary Use Case và các nhóm sinh viên có khác biệt về kết quả quan sát được không? | 7, 8 |
| RQ06 | AI Dependency có liên hệ với GPA, GPA Change và Skill Retention không? | 7, 8 |
| RQ07 | Có dấu hiệu phi tuyến, bão hòa hoặc tương tác giữa các đặc trưng không? | 8–10 |
| RQ08 | Nhóm AI features có cải thiện dự đoán so với baseline học tập không? | 9, 12 |
| RQ09 | Có đa cộng tuyến hoặc phụ thuộc tuyến tính hoàn hảo không? | 9, 10 |
| RQ10 | Nên chọn preprocessing, encoding, scaling, feature engineering nào? | 3, 9, 10, 12 |
| RQ11 | Mô hình tuyến tính có thể gặp giới hạn hoặc sai lệch nào? | 6, 9, 11 |
| RQ12 | Nguồn gốc và tính đại diện của dữ liệu giới hạn khả năng khái quát ra sao? | 11, 12 |

## 3. Kiến trúc thực thi và quản lý dữ liệu

### 3.1. Pipeline

```text
Raw CSV (read-only)
  -> load / schema validation / structural audit
  -> deterministic validity rules / duplicate policy / stable row keys
  -> create & freeze train / validation / test assignments
  -> detailed EDA on TRAIN only
  -> feature engineering exploration + train-only fitting
  -> exploratory regression + residual diagnostics (train)
  -> limited exploratory comparison on VALIDATION
  -> evidence-based findings + preprocessing/model recommendations
  -> export tables, figures, reports and manifest
  -> TEST remains untouched for final evaluation outside EDA
```

- Structural audit trên toàn bộ file chỉ gồm kiểm tra schema, tính toàn vẹn, quy tắc hợp lệ có căn cứ độc lập, missing/duplicate counts và thông tin cần thiết để chia tập. Phải gắn nhãn `full_structural_audit`.
- Không dùng phân phối target toàn dataset, correlation toàn dataset, outlier thresholds toàn dataset hoặc kết quả test để ra quyết định mô hình.
- Các mục chi tiết từ descriptive statistics trở đi dùng train. Validation chỉ cho so sánh thăm dò đã định trước; không tối ưu nhiều lần trên validation. Không tính test metrics, test target distributions hoặc test-derived recommendations trong notebook.
- Default split 70/15/15, seed 42; cấu hình được. Nếu dữ liệu có nhiều dòng/sinh viên, giữ nhóm Student_ID cùng split. Các bản ghi xác nhận trùng phải xử lý trước split; bản ghi giống nhau ngoài ID nhưng chưa xác nhận trùng cần đánh giá và group khi có nguy cơ leakage, không tự ý xóa sinh viên khác nhau. Nếu dùng group split, báo cáo tỷ lệ thực tế và không buộc khớp tuyệt đối tỷ lệ mục tiêu.
- Dùng stable row key dựa trên phiên bản/fingerprint dữ liệu và vị trí dòng gốc hoặc ID duy nhất đã xác minh; không dùng Python hash ngẫu nhiên giữa các tiến trình.
- Dòng thiếu/không hợp lệ ở target: không impute target để train; ghi lại lý do loại khỏi tập hồi quy. Missing predictors và thống kê ngưỡng outlier học từ dữ liệu được xử lý sau split, fit trên train. Không tự động xóa outlier hợp lệ.
- Nếu chọn stratification theo quantile của target để tạo split, ghi rõ đây chỉ là thao tác phân chia, không phân tích/điều chỉnh test và không dùng bins làm predictor. Mặc định random/group split để đơn giản.
- Raw CSV không được sửa. Lưu fingerprint SHA-256, row counts, split seed, split rules, exclusions, path/phiên bản dataset và library versions để tái lập.

### 3.2. Centralized Configuration

Một cell cấu hình duy nhất ở đầu notebook (không phụ thuộc đường dẫn cá nhân):

| Key | Default/behavior |
|---|---|
| `DATA_PATH` | Cấu hình đường dẫn CSV; tự tìm ứng viên nếu chưa đặt; nếu nhiều file hợp lệ thì báo rõ để chọn |
| `OUTPUT_DIR` | `reports/` |
| `SPLIT_DIR` | `data/splits/` |
| `TARGET_COLUMN` / `ID_COLUMN` | `Post_Semester_GPA` / `Student_ID` |
| `RANDOM_STATE` | 42 |
| `TRAIN_RATIO`, `VALIDATION_RATIO`, `TEST_RATIO` | 0.70, 0.15, 0.15; kiểm tra tổng bằng 1 |
| `PLOT_SAMPLE_SIZE` | 5.000 (chỉ visualization khi cần) |
| `FIGURE_DPI` | 300 |
| `SAVE_FIGURES`, `SAVE_TABLES` | True |
| `RUN_OPTIONAL_TESTS` | False hoặc cấu hình rõ |
| `MIN_GROUP_SIZE` | 30 (ngưỡng cảnh báo diễn giải nhóm nhỏ, có thể điều chỉnh) |
| `GPA_EQUAL_TOLERANCE` | Suy ra từ độ chính xác đã xác minh hoặc dùng tolerance floating-point nhỏ và ghi rõ |
| `PREDICTION_CUTOFF` | Trước khi biết GPA cuối kỳ; cần ghi rõ giả định về thời điểm đo từng feature |

Nếu tỷ lệ split hay fingerprint khác cấu hình đã lưu, không âm thầm tái sử dụng/ghi đè split; tạo run mới hoặc thông báo thay đổi rõ ràng.

## 4. Notebook structure và yêu cầu phân tích

Notebook `notebooks/01_EDA.ipynb` gồm 12 sections. Có thể thêm cell configuration/imports/helpers trước Section 1. Mỗi subsection có Markdown objective/methodology, code, output thực tế, interpretation, modeling implication và limitation khi có. Tiêu đề, plot labels, table columns và code ưu tiên English; nhận xét học thuật bằng tiếng Việt.

Mã `T01–T48`, `F01–F57` là danh mục coverage dự kiến, không phải quota tạo hình/bảng trùng. Artifact tương đương có thể dùng chung file nếu manifest ghi rõ alias; nếu không áp dụng, ghi `skipped` kèm lý do cụ thể. Không đánh dấu generated nếu file chưa tồn tại.

### Section 1 — Introduction (P0)

Trình bày problem statement, motivation, research questions (RQ01–RQ12), data source, target, forecast scenario, giới hạn nguồn dữ liệu và phạm vi EDA. Giải thích tính phù hợp với Linear Regression và phân biệt association/prediction/causation.

### Section 2 — Dataset Overview (P0)

- Load dữ liệu thực tế; kiểm tra path, encoding, header, required và extra columns; báo cáo mọi mapping/rename, không sửa âm thầm.
- `shape`, `head/sample`, `info`, dtype, memory usage, unique count, semantic types, missingness và feature roles.
- Dataset có schema khác dự kiến: thiếu target hoặc không thể xác định nhóm cốt lõi thì fail rõ ràng; thiếu optional column thì chỉ bỏ qua phân tích phụ thuộc vào nó. Extra columns được mô tả, mặc định chưa được đưa vào model.
- Hiển thị sample an toàn, không xuất danh sách ID cá nhân không cần thiết.

**Tables:** T01 Dataset Overview; T02 Data Dictionary; T03 Feature Type Summary; T04 Unique Values Summary.

**Figures:** F01 Feature Types Distribution (bar chart).

### Section 3 — Data Quality Assessment (P0)

- Missing: count, %, NaN, empty/whitespace và sentinel values có bằng chứng; không đánh đồng 0 với missing.
- Duplicates: exact rows, duplicate Student_ID, duplicate excluding ID, same ID/conflicting information. Phân biệt số nhóm và số dòng ảnh hưởng.
- Dtypes: numeric strings, inconsistent boolean/categories, case/space differences. Không ép kiểu làm mất thông tin mà không log.
- Validity: GPA theo thang điểm được xác minh, thời gian không âm, counts hợp lệ, finite numbers, ordinal domains được xác minh. Không suy ra quy tắc hợp lệ chỉ từ observed min/max.
- Outliers (train): Q1/Q3/IQR, fences 1.5×IQR, count/%; Z-score bổ sung khi phù hợp; quantiles và boxplots. Với IQR=0, constant hoặc sparse discrete feature, ghi rõ hạn chế và không flag máy móc.
- Data quality decision: feature/issue, severity, affected rows, rule, action (`retain`, `correct`, `remove`, `investigate`), lý do, data scope. Báo cáo before/after count và exclusions. Nếu không có lỗi, vẫn ghi kết quả 0 hoặc trạng thái passed.

**Tables:** T05 Missing Values Summary; T06 Duplicate Records Summary; T07 Invalid Values Summary; T08 Outlier Summary; T09 Data Quality Report.

**Figures:** F02 Missing Percentage; F03 Missing Pattern (nếu có missing); F04 Numerical Boxplots; F05 Outlier Percentage by Feature.

### Section 4 — Descriptive Statistics (P0)

Trên train:

- Numerical: valid count, missing %, mean, median, mode nếu có ý nghĩa, sample std/variance (ddof=1), min/max, range, Q1/Q3, IQR, skewness, kurtosis (nêu convention), quantiles.
- Categorical/ordinal/binary: unique count, categories, frequency, percentage, mode, least frequent, nhóm hiếm, missing group. Ghi rõ mẫu số percentage.
- Không dùng mean cho nominal features đã mã hóa tùy ý. Không suy ra normality requirement cho predictors.

**Tables:** T10 Numerical Descriptive Statistics; T11 Categorical Descriptive Statistics; T12 Category Frequency Distribution; T13 Distribution Characteristics and Skewness.

### Section 5 — Univariate Analysis (P0)

- Continuous: histogram + KDE nếu phù hợp, boxplot, distribution shape, center, dispersion, skewness, extreme values.
- Discrete/ordinal/binary: ordered frequency bar plots; không áp dụng KDE không phù hợp. Nominal: count plots, label readable, group sizes.
- Bao phủ toàn bộ thuộc tính ngoài ID; target và auxiliary outcomes phải được nhận diện đúng vai trò. ID chỉ audit tính duy nhất.
- Mỗi feature có interpretation và implication: missing handling, encoding, transformation candidate, outlier retention hoặc investigation. Không loại biến chỉ vì phân phối lệch.

**Tables:** tái sử dụng T10–T13.

**Figures:** F06 Numerical Histograms; F07 Continuous Density Plots; F08 Numerical Boxplots (có thể alias F04); F09 Categorical Count Plots; F10 Ordinal Distributions; F11 Skewness Comparison.

### Section 6 — Target Variable Analysis (P0)

- Post GPA: distribution, summary, min/max, boundary counts và tỷ lệ ceiling/floor effect theo thang điểm đã xác minh; không giả định thang 4.0 nếu chưa xác minh.
- GPA Change: tính trên các dòng có cả hai GPA hợp lệ; histogram, mean/median/std, improved/unchanged/declined với tolerance có giải thích; ghi sample count và denominator.
- Paired Pre/Post comparison: cùng rows, mean/median difference, Pearson/Spearman, scatter/hexbin với identity line `y=x` và cùng axis scale.
- GPA Change theo Pre GPA groups: bins có căn cứ, count, mean và uncertainty nếu phù hợp; lưu ý mathematical coupling, regression to the mean, ceiling effects.
- Nhấn mạnh GPA change không đồng nghĩa tác động AI.

**Tables:** T14 Target Statistics; T15 GPA Change Statistics; T16 Improvement Summary; T17 GPA Change by Initial GPA Group; T18 GPA Boundary Concentration.

**Figures:** F12 Post GPA Distribution; F13 Pre vs Post GPA Distribution; F14 GPA Change Distribution; F15 Improved/Unchanged/Declined; F16 Pre vs Post GPA with Identity Line; F17 GPA Change by Initial GPA Group.

### Section 7 — Bivariate Analysis (P0)

#### Numerical / ordinal vs target

- Với các biến phù hợp: so sánh với Post GPA và GPA Change; Pearson cho quantitative variables, Spearman cho quan hệ monotonic và ordinal có thứ tự được xác minh. Không gán số tùy ý cho nominal để tính Pearson.
- Báo cáo hệ số, valid pair count, dấu, độ lớn; constant/missing/insufficient pairs trả N/A có lý do.
- Predictor vs target scatter/hexbin, linear fit và LOWESS hoặc binned smooth cho AI Hours. Sử dụng sampling tái lập nếu cần hiển thị, thống kê chính tính trên toàn train hợp lệ.
- Phân biệt marginal association với conditional association; correlation yếu không tự động dẫn đến loại feature.
- Auxiliary outcomes được phân tích riêng có nhãn, không đưa vào predictor-only matrix.

#### Categorical vs target

- Major, Year, Primary Use Case, Prompt Skill, Paid Subscription, Institutional Policy; và các nhóm phù hợp khác có trong schema.
- Count, mean/median/std Post GPA, mean GPA Change, uncertainty (ví dụ 95% CI nếu hợp lý); box/violin hoặc point plot theo nhóm. Nhóm nhỏ phải gắn cảnh báo.
- Optional ANOVA/Kruskal–Wallis khi assumptions phù hợp, effect size và multiple-comparison correction nếu làm nhiều test. Không dùng p-value làm tiêu chí duy nhất.

**Tables:** T19 Pearson with Target; T20 Spearman with Target; T21 Categorical vs Target Summary; T22 Group Difference Summary.

**Figures:** F18 Numerical Predictors vs Post GPA; F19 Numerical Predictors vs GPA Change; F20 Pearson Heatmap; F21 Spearman Heatmap; F22 Feature Correlation with Post GPA; F23 Feature Correlation with GPA Change; F24 AI Hours vs GPA (linear + nonlinear trend); F25 Post GPA by Major; F26 by Year; F27 by Primary Use Case; F28 by Prompt Skill; F29 by Paid Subscription; F30 by Institutional Policy; F31 GPA Change by Categorical Groups.

### Section 8 — Multivariate Analysis (P1)

So sánh mối liên hệ có điều kiện, các subgroup và giả thuyết interaction:

| Kết hợp | Đầu ra mong muốn |
|---|---|
| AI Hours × Prompt Skill | Facet/grouped trend với count và GPA/Change |
| AI Hours × Major | Facet trend theo ngành |
| AI Hours × Traditional Study Hours | Binned 2D heatmap: mean target + count |
| AI Hours × Pre GPA | Grouped trend theo GPA ban đầu |
| AI Hours × AI Dependency | Group summary hoặc adjusted relationship |
| Primary Use Case × Prompt Skill | Heatmap mean GPA/Change + count |
| AI Dependency × Exam Anxiety | Heatmap mean GPA/Change + count |
| AI Dependency / AI Hours × Skill Retention | Auxiliary outcome association |

- Bins cho biến continuous: xác định trên train theo domain hoặc quantiles, ghi edges và cách xử lý ties/duplicate edges. Không dùng bins để áp đặt trước xu hướng.
- Báo cáo size từng nhóm/cell, suppress hoặc flag cell ít mẫu; không tô 0 thay cho missing cell. So sánh có kiểm soát Pre GPA hoặc dùng adjusted residual/OLS khi thích hợp; không gọi là causal moderation.

**Tables:** T23 AI Usage Intensity Groups; T24 Academic/AI Group Summary; T25 Interaction Analysis Summary; T26 Subgroup Summary.

**Figures:** F32 GPA Change by AI Usage Hours; F33 AI Hours vs GPA by Prompt Skill; F34 AI Hours vs GPA by Major; F35 AI Hours × Traditional Study Heatmap; F36 Use Case × Prompt Skill Heatmap; F37 Dependency × Anxiety Heatmap; F38 AI Hours vs GPA Change by Pre GPA; F39 AI Hours vs Skill Retention (có thể kèm Dependency panel).

### Section 9 — Regression-Oriented Analysis (P0)

#### 9.1. Linearity

- Linear fit vs LOWESS hoặc binned trend; so sánh Pearson/Spearman.
- Kiểm tra quan hệ có điều kiện bằng residual plots hoặc partial relationship sau khi fit exploratory OLS.
- Ghi nhận gần tuyến tính, yếu, nonlinear, threshold hoặc saturation nếu có bằng chứng; đề xuất transformation để kiểm chứng tiếp, không khẳng định trước.

#### 9.2. Multicollinearity

- Predictor-only correlation matrix, high-correlation pair list (threshold cấu hình được, ví dụ |r| ≥ 0.8 chỉ để flag).
- VIF trên design matrix fit từ train: impute/encode hợp lệ, reference category/drop-one cho OLS có intercept, loại constant columns có log; không có target, ID hoặc auxiliary outcomes bị cấm. Tính VIF với intercept nhất quán, không báo VIF intercept như feature.
- Báo rank, số cột, singularity, condition number (ghi scale); VIF <5 thấp, 5–10 cần xem xét, >10 cao (heuristics, không phải quy tắc xóa).
- Kiểm tra AI Hours–Dependency, Dependency–Anxiety nếu được phép; H và H², dummy trap, total hours và thành phần. Nếu rank deficient, ghi dependencies/infinite VIF; không xem hệ số không định danh là kết luận đáng tin.

#### 9.3. Exploratory Regression

| ID | Model | Input / purpose |
|---|---|---|
| E0a | Mean baseline | Mean train target |
| E0b | Persistence baseline | Pre GPA làm dự đoán Post GPA nếu có |
| E1 | Simple OLS | Pre GPA |
| E2 | Academic OLS | E1 + eligible academic variables |
| E3 | Academic + AI OLS | E2 + eligible AI features; nếu timing chưa xác minh, chạy riêng như retrospective/conditional comparison |

- Fit trên train, preprocessing train-only; so sánh trên cùng validation rows hoặc cùng folds nếu thực hiện CV nội bộ. Báo MAE, MSE, RMSE, R², sample count; adjusted R² chỉ trên OLS fitted sample với điều kiện hợp lệ. R² âm là kết quả hợp lệ.
- Báo delta MAE/RMSE/R² giữa E1/E2/E3, phân biệt giá trị bổ sung của academic và AI features. Nếu các biến chưa xác minh làm models khác prediction scenario, gắn nhãn rõ, không so sánh như cùng bài toán triển khai.
- Bảng coefficients: feature, reference group, coefficient, đơn vị, confidence interval/SE khi phù hợp, dấu, model/scenario. Không diễn giải hệ số là tác động nhân quả; không so độ lớn raw coefficients khác đơn vị như importance.
- Không tuning cuối cùng, không evaluate test, không ghi kết quả validation là final test performance.

#### 9.4. Residual Diagnostics (train fitted models)

- Residual = actual − fitted; actual vs predicted + identity line, residual vs fitted + zero line, residual vs key predictors, histogram và Q-Q, scale-location. Báo prediction ngoài thang GPA nếu thang được xác minh, không âm thầm clip dự đoán.
- Kiểm tra linearly specified mean, homoscedasticity, residual normality cho inference, independence dựa vào thiết kế dữ liệu/grouping. Không suy ra independence từ residual plot; predictor normality không phải yêu cầu OLS.
- Optional Breusch–Pagan, Jarque–Bera, robust SE (HC3) nếu phù hợp. Với 50k mẫu, ưu tiên effect size/plot và giới hạn thực tiễn thay vì chỉ nhìn p-value.

#### 9.5. Influential Observations

- Standardized hoặc studentized residual (ghi loại), leverage, Cook's distance. Dùng công thức hoặc statsmodels phù hợp quy mô; không tạo hat matrix N×N không cần thiết.
- Các ngưỡng |residual|>3, leverage>2p/n, Cook's D>4/n chỉ là heuristics; ghi n, p và phương pháp; không tự động xóa. Báo top influential rows bằng stable keys không phơi ID không cần thiết; sensitivity analysis là tùy chọn trên train.

**Tables:** T27 Highly Correlated Predictor Pairs; T28 VIF Analysis (kèm rank/condition); T29 Exploratory Regression Results; T30 Coefficients; T31 Regression Assumption Diagnostics; T32 Influential Observations Summary.

**Figures:** F40 Predictor Correlation Heatmap; F41 VIF Comparison; F42 Actual vs Predicted GPA; F43 Residual vs Fitted; F44 Residual Distribution; F45 Q-Q Plot; F46 Scale-Location; F47 Cook's Distance; F48 Residuals vs Key Predictors; F49 Exploratory Model Comparison.

### Section 10 — Feature Engineering Exploration (P1)

| Candidate | Definition | Constraint |
|---|---|---|
| `GPA_Change` | Post GPA − Pre GPA | Analytical target only; forbidden predictor |
| `Total_Study_Hours` | AI Hours + Traditional Study Hours | Chỉ nếu đơn vị, thời gian và phạm vi học tập tương thích, không double-count |
| `AI_Study_Ratio` | AI Hours / Total Study Hours | Chỉ nếu tổng hợp lệ; denominator=0 -> missing/flag, không gán 0 tùy ý |
| `AI_Hours_Squared` | H² | Candidate phi tuyến |
| `Log_AI_Hours` | log1p(H) | H không âm đã xác minh |
| `Centered_AI_Hours` | H − mean_train(H) | Center fit trên train |
| `Centered_AI_Hours_Squared` | (H − mean_train(H))² | Giữ lower-order term khi dùng polynomial |
| `AI_Dependency_Interaction` | H × Dependency | Chỉ khi ordinal scale và availability phù hợp |
| `AI_Prompt_Interaction` | H × encoded Prompt Skill | Prefer reference-coded dummy interactions; ordinal chỉ khi giả định hợp lý |
| `Study_Anxiety_Interaction` | Study Hours × Anxiety | Conditional/retrospective nếu chưa xác minh timing |

- Không mặc định mọi AI Hours là study hours; nếu điều kiện không xác minh được thì skip total/ratio và ghi lý do.
- So distribution, skewness, target correlation, redundancy/VIF, coefficients và limited validation comparison nếu cần; không tự động thêm toàn bộ feature candidates.
- Tránh raw H, centered H cùng intercept; tránh Total + hai thành phần cùng lúc; tuân thủ hierarchy khi thêm interactions/polynomials.
- Encoding: nominal one-hot; ordinal one-hot hoặc ordered numerical có giải thích khoảng cách; binary mapping theo categories thực tế; unknown categories handled; ID excluded. Imputation và encoding fit train; log/centering/scaling có domain check.
- Scaling: OLS không bắt buộc, chuẩn hóa để so sánh coefficient nếu phù hợp; polynomial và regularized regression (Ridge/Lasso) nên scale; không fit scaler trên validation/test.

**Tables:** T33 Derived Feature Definitions; T34 Derived Feature Statistics; T35 Original vs Transformed Comparison; T36 Encoding Recommendations; T37 Scaling Recommendations; T38 Feature Selection Recommendations.

**Figures:** F50 Original vs Transformed AI Hours; F51 Derived Feature Distributions; F52 Derived Feature Correlations; F53 Linear vs Polynomial Relationship; F54 Correlation Before/After Engineering.

### Section 11 — Data Leakage & Bias Analysis (P0)

Phần audit được tổng hợp ở đây nhưng quy tắc bảo vệ dữ liệu phải được áp dụng trước Section 4 và trước khi fit bất kỳ model nào.

- Target leakage: target/GPA Change không được đưa vào X; ID dùng cho audit/grouping; Skill Retention và Burnout mặc định excluded vì là auxiliary outcomes chưa xác minh thời điểm.
- Temporal leakage: matrix mỗi feature có availability (`available`, `conditional`, `unavailable`, `unknown`), evidence, cutoff, risk, decision. Những feature AI/hành vi chưa rõ timing chỉ vào retrospective analysis có nhãn, không vào predictive scenario chính.
- Split leakage: row-key disjointness, Student_ID/group overlap, duplicate-content overlap, split assignment fingerprint, train-only fit. Không test target profiling.
- Bias/provenance: synthetic status/license/version nếu có evidence, sampling design/representativeness, group imbalance, rounding/clipping, near-deterministic relations, limited external validity. Không suy diễn cơ chế sinh dữ liệu thành sự thật.
- Nếu dataset synthetic được xác minh, mô hình chỉ học quan hệ trong dữ liệu mô phỏng; không mặc định generalize đến sinh viên thật và không khẳng định AI gây thay đổi GPA.

**Tables:** T39 Leakage Risk Assessment; T40 Feature Availability Matrix; T41 Split Integrity Report; T42 Dataset Bias and Limitations.

**Figures:** F55 Sample Counts Across Splits (không target); F56 Target Distribution Train vs Validation (optional audit; không test); F57 Feature Availability and Leakage Risk Matrix.

### Section 12 — EDA Findings & Recommendations (P0)

Tổng hợp findings truy vết được và trả lời RQ01–RQ12. Mỗi finding có ID `FND-xxx`, RQ, observation, số liệu tính trực tiếp, data scope, table/figure IDs, interpretation, recommended action/model implication, limitation. Nếu thiếu bằng chứng, đánh dấu unresolved/not applicable; không tự tạo nhận xét định lượng.

Feature decision matrix gồm: feature, semantic type, availability, missing %, Pearson/Spearman khi hợp lệ, VIF khi hợp lệ, leakage risk, encoding, transformation, decision (`Keep`, `Transform`, `Conditional`, `Exclude`, `Investigate`), lý do, evidence IDs. Có thể báo VIF của từng encoded column/term và tổng hợp minh bạch theo feature gốc; không gán tùy ý một điểm feature importance.

**Tables:** T43 Research Questions and Evidence; T44 Key EDA Findings; T45 Feature Decision Matrix; T46 Preprocessing Plan; T47 Linear Regression Model Recommendations; T48 Dataset Limitations.

Mọi finding quan trọng phải nối được chuỗi: **Research question → observed evidence → interpretation → modeling decision**.

## 5. Model recommendations for subsequent training

EDA phải cung cấp evidence cho các model candidates dưới đây, nhưng không bắt buộc triển khai full training trong notebook. Số mô hình cuối cùng phải tối thiểu bằng số thành viên theo Lab 01; danh sách 5 mô hình là thiết kế đề xuất, cần điều chỉnh theo nhóm và bằng chứng.

| ID | Model | Candidate features | Câu hỏi đánh giá |
|---|---|---|---|
| M01 | Simple Linear Regression | Pre GPA | Baseline học tập mạnh đến đâu? |
| M02 | Multiple Linear Regression — Academic | M01 + eligible Major, Year, Traditional Study Hours | Academic features có cải thiện dự đoán? |
| M03 | Multiple Linear Regression — Academic + AI | M02 + eligible AI Hours, Use Case, Prompt Skill, Tools, Paid Subscription, các biến hợp lệ khác | AI features có giá trị bổ sung không? |
| M04 | Polynomial Linear Regression | M03 + centered H² và selected interaction terms có bằng chứng | Quan hệ phi tuyến/tương tác có cải thiện validation performance? |
| M05 | Ridge Regression | Eligible features đã encode/scale, feature set xác định từ EDA | Regularization có giúp ổn định hệ số và generalization? |

- Nếu AI features chưa xác minh temporal availability, M03–M05 sử dụng chúng chỉ là candidates cho retrospective scenario; không tuyên bố chính xác như forecast. M05 có thể dùng cùng eligible feature set của M02 hoặc M03 tùy scenario; so sánh phải ghi rõ feature set.
- M04 vẫn là mô hình tuyến tính theo coefficients dù sử dụng feature phi tuyến. Ridge là linear model có L2 regularization; xác minh phạm vi phương pháp được chấp nhận theo quy định môn học trước khi chọn làm một trong số mô hình bắt buộc.
- Metrics chính: MAE, MSE, RMSE, R²; dùng cùng target, splits, rows và evaluation protocol. Baseline mean/persistence nên được giữ làm reference. Không ưu tiên chỉ R² nếu incremental performance nhỏ.
- Sau EDA, lựa chọn biến/transformations/model trên train + validation hoặc CV; final test chỉ dùng sau khi quyết định đã đóng băng. Báo riêng extrapolation/prediction ngoài domain nếu có.

## 6. Deliverables và cấu trúc lưu trữ

```text
project/
├── data/
│   ├── raw/                         # CSV gốc, read-only, không copy bắt buộc
│   └── splits/
│       ├── split_assignment.csv     # row_key, split, exclusion/group info phù hợp
│       └── split_metadata.json      # fingerprint, seed, ratios, rules
├── notebooks/
│   └── 01_EDA.ipynb                  # deliverable chính, code + markdown + outputs
├── reports/
│   ├── EDA_FINDINGS.md
│   ├── DATA_DICTIONARY.md
│   ├── PREPROCESSING_PLAN.md
│   ├── artifact_manifest.csv
│   ├── run_metadata.json
│   ├── figures/                     # F01_*.png ... F57_*.png
│   └── tables/                      # T01_*.csv ... T48_*.csv
├── requirements.txt
└── README.md                       # cách chạy hoặc cập nhật docs sẵn có
```

Tên và vị trí có thể theo conventions của repository nếu đã có layout tương đương. Không ghi đè notebook, README hay dependency file không liên quan; ưu tiên tích hợp thay vì tạo nhiều cấu trúc trùng.

### 6.1. Notebook và reports

- `01_EDA.ipynb`: notebook Jupyter hợp lệ, markdown + executable code + inline tables/plots. Khi có CSV thực tế, lưu một bản executed notebook với outputs thật; notebook là nguồn tái tạo artifacts và reports.
- `EDA_FINDINGS.md`: data version/scope, summary, RQ mapping, quantitative evidence với references đến artifact IDs, implications, limitations, unresolved issues. Có đường dẫn tương đối tới bảng/hình liên quan. Không viết số liệu trước khi có output.
- `DATA_DICTIONARY.md`: observed schema, feature semantics, units, domain checks, role, timing/provenance và unknowns.
- `PREPROCESSING_PLAN.md`: từng feature/issue, action, rationale, fit scope, leakage restrictions, encoding, transformations, outlier policy và candidate model dependencies. Phân biệt quyết định đã thực hiện trong EDA và đề xuất cho training.
- `split_assignment.csv` không cần chứa target hay toàn bộ dữ liệu gốc; lưu stable row key, split, group key nếu cần và phù hợp, exclusion status/reason nếu có. Đảm bảo có thể map lại dataset bằng fingerprint và key.
- `run_metadata.json`: dataset hash, actual path/version, config, Python/library versions, run status, row counts, split counts, warnings, unresolved verification issues.

### 6.2. Artifact manifest

CSV có một record cho từng artifact ID dự kiến:

| Field | Meaning |
|---|---|
| `artifact_id` | T01–T48 / F01–F57 |
| `artifact_type` | table / figure |
| `title` | Descriptive title |
| `relative_path` | Relative path tới file thực sự được tạo, trống nếu skipped |
| `section` | Notebook section |
| `data_scope` | full_structural_audit / train / validation / split_metadata |
| `status` | generated / reused / skipped / failed |
| `related_ids` | ID artifact dùng chung file hoặc liên quan |
| `reason` | Skip/failure/alias explanation |

Một run đầy đủ phải cover toàn bộ ID bằng trạng thái trung thực; file absent không được ghi generated. `failed` không thay thế `skipped` đối với lỗi thực thi bất ngờ. Manifest phải nhất quán với artifact files, không bao gồm artifacts cũ từ lần chạy trước như kết quả mới. Nếu save được tắt bằng config, ghi skip reason và notebook vẫn hiển thị output.

## 7. Implementation guidelines

### 7.1. Environment và thư viện

Hỗ trợ CPU trên Kaggle, Colab, Jupyter local và VS Code trên Windows; không yêu cầu GPU, API trả phí, LLM hoặc mạng trong lúc chạy nếu CSV/dependencies đã sẵn có.

- `pandas`, `numpy`: load, summarize, clean, aggregate.
- `matplotlib`, `seaborn`: plots, consistent styles, export.
- `scipy`: correlations, optional tests.
- `scikit-learn`: splits, pipelines, encoding, scaling, exploratory regression, metrics.
- `statsmodels`: OLS, VIF, residual diagnostics, influence analysis.
- `pathlib`, `hashlib`, `json`: paths, metadata, fingerprints.
- `nbformat`, `nbclient` hoặc `jupyter nbconvert`: notebook validation và execution test.

Dùng relative paths hoặc absolute paths từ configuration; không hardcode drive cá nhân, Kaggle input hash hoặc cố định filename chưa xác minh. Không tự động cài dependencies trong mỗi lần chạy. `requirements.txt` phải có version constraints phù hợp runtime đã kiểm thử.

### 7.2. Notebook code quality

- Chạy bằng Restart & Run All từ kernel sạch; không hidden state, không phụ thuộc thứ tự cell thủ công, không manual copy số từ output.
- Tái sử dụng helpers, ví dụ `save_table`, `save_figure`, `register_artifact`, `validate_schema`, `summarize_numeric`, `summarize_categories`, `safe_correlation`, `audit_splits`. Tên hàm có thể thay đổi theo thiết kế.
- Helpers export phải tạo parent dirs, ghi path bằng manifest, xử lý save flags, đóng figure sau khi hiển thị/lưu để tiết kiệm bộ nhớ.
- Dùng `ColumnTransformer`/`Pipeline` hoặc logic tương đương có train-only fit; preserve encoded feature names, reference levels và biến bị loại để giải thích coefficients/VIF. Optional analysis thiếu cột phải skip có lý do.
- Không broad `except` để bỏ qua lỗi bất ngờ, không suppress toàn bộ warnings, không thay NaN bằng 0 tùy tiện.
- Sử dụng vectorized operations, tránh N×N structures và các scatter/LOWESS/influence computations quá nặng. Nếu subsample cho trend/visualization, ghi sample size/seed, giữ computed statistics trên toàn bộ subset hợp lệ được phép.
- Notebook có thể chạy lại với cùng seed/config mà không nhân bản artifacts; không xóa dữ liệu của người dùng. Quản lý output theo run hoặc clear chỉ những generated artifacts do notebook sở hữu.

### 7.3. Tables và visualization standards

**Tables:** actual values, IDs, dễ đọc, rõ denominator và valid n, đơn vị và data scope; rounding chỉ khi display; CSV UTF-8 với precision đầy đủ, NaN/N/A khi không áp dụng và có lý do. Bảng dài có thể hiển thị preview + lưu toàn bộ.

**Figures:** generated từ dữ liệu thật, title, axis labels/units, legends, readable font, consistent palette, adequate spacing, 300 DPI PNG; SVG tùy chọn. Heatmap có color scale và counts khi aggregate. Histogram dùng bins nhất quán khi so sánh, scatter dùng sampling/alpha/hexbin; categorical labels không bị cắt. CI/error bars có mô tả phương pháp; không dùng biểu đồ 3D hay axis gây hiểu nhầm.

Mỗi bảng/hình có caption hoặc interpretation liên quan câu hỏi nghiên cứu. Một figure có thể chứa nhiều subplots hợp lý; không vẽ lại cùng phân tích chỉ để tăng số lượng. Không dùng hình minh họa synthetic như bằng chứng thực tế.

## 8. Edge cases và failure policy

| Case | Expected behavior |
|---|---|
| CSV không tồn tại hoặc nhiều ứng viên | Báo path/ứng viên rõ ràng, yêu cầu cấu hình; không tự tạo dữ liệu giả |
| Thiếu required target, dữ liệu rỗng hoặc không đủ train để fit | Fail fast với actionable message |
| Thiếu optional columns hoặc extra columns | Audit schema, skip phân tích phụ thuộc, mô tả extra columns |
| Dtype/category không đồng nhất | Log mapping và conversion failures; không silent coercion |
| NaN, Inf, sentinel, target invalid | Áp dụng quy tắc có căn cứ; lưu count/reason; không impute target |
| Constant feature, one category, IQR=0 | N/A/skip method không có ý nghĩa; report rõ |
| Rare categories hoặc subgroup nhỏ | Count + caution; không tự suy diễn hoặc merge không có lý do |
| Correlation undefined | N/A và valid pair count, không gán 0 |
| Singular matrix / infinite VIF | Report dependencies; giải quyết reference/constant/redundancy có log trước inference |
| Total hours=0 / unit unknown | Ratio undefined hoặc skip; không tạo các giả định về thời gian học |
| GPA precision/timing/provenance unknown | Ghi uncertainty; giới hạn phân tích/mô hình tương ứng |
| Duplicate/group split constraints làm lệch ratios | Báo actual ratios, ưu tiên không overlap |
| Optional statistical test không hợp lệ | Skip với reason; core analysis vẫn chạy |
| Test subset | Chỉ metadata/counts và integrity; không metrics, target distribution, tuning |
| Output directory đã tồn tại | Không ghi đè unrelated files hoặc gán artifacts cũ cho current run |

## 9. Acceptance criteria và verification checklist

Notebook hoàn thành khi các tiêu chí sau được kiểm tra thực tế:

- [ ] AC01 — File `notebooks/01_EDA.ipynb` hợp lệ theo nbformat, có đủ 12 sections và centralized configuration.
- [ ] AC02 — Dataset source/schema/row count được xác minh từ CSV thực tế; tất cả cột được mô tả, các khác biệt với schema dự kiến có audit.
- [ ] AC03 — Structural audit bao phủ missing, duplicates, invalid values, dtype; data cleaning decisions có evidence và counts trước/sau.
- [ ] AC04 — Train/validation/test được tạo và lưu ổn định; không row/group overlap bị cấm; fingerprint raw dataset không thay đổi.
- [ ] AC05 — Descriptive statistics và univariate analysis bao phủ toàn bộ applicable features; ID không bị dùng sai.
- [ ] AC06 — Post GPA, paired GPA comparison, GPA Change, improved/unchanged/declined, ceiling/floor được phân tích hoặc skip với reason xác minh.
- [ ] AC07 — Bivariate analysis có Pearson/Spearman, group statistics, plots, valid sample counts; không tính Pearson từ nominal encoding tùy ý.
- [ ] AC08 — Multivariate/interaction analysis bao phủ các cặp áp dụng được, có counts, bins/threshold rõ và cautious interpretation.
- [ ] AC09 — Regression-oriented section có predictor-only correlation, VIF, rank, exploratory models, metrics, coefficients, residual và influence diagnostics hợp lệ.
- [ ] AC10 — Feature engineering có formulas, domain checks, transformation comparisons, recommendation dựa trên observed evidence và leakage assessment.
- [ ] AC11 — Feature availability, synthetic/provenance limitations, bias và test-set policy được ghi rõ; không dùng target-derived feature hoặc ID làm predictors.
- [ ] AC12 — Tất cả planned artifact IDs trong manifest có status đúng; mọi generated/reused path tồn tại và file đọc được; tất cả applicable exports được tạo nếu flags bật.
- [ ] AC13 — EDA findings có IDs, quantitative evidence đúng output, RQ mapping, limitations và model implications; không có manufactured statistics, metrics hoặc causal claims.
- [ ] AC14 — Data Dictionary, Preprocessing Plan, Findings, manifest, split metadata và run metadata tồn tại khi full run thành công.
- [ ] AC15 — Restart & Run All hoặc execution bằng nbclient/nbconvert thành công trên CSV thật với dependencies hợp lệ; không unhandled errors.
- [ ] AC16 — Chạy lại cùng data/config/seed cho cùng split assignments và các thống kê chính (so sánh số với tolerance phù hợp); fingerprint dataset không đổi.
- [ ] AC17 — Test set không được dùng cho các phân tích/pipeline fit/model selection; audit code và data scope của artifacts để kiểm chứng.
- [ ] AC18 — Đề xuất model candidates có rationale, eligible features, scenario và evidence; phân biệt exploratory validation với final evaluation.

### 9.1. Minimum validation procedure

1. Kiểm tra bố cục repo, dependency và các file tồn tại; xác định dataset file thực tế.
2. Validate notebook schema bằng `nbformat.validate`.
3. Thực thi notebook trong kernel sạch bằng nbclient hoặc nbconvert, với CSV thực tế được gắn vào configuration. Ví dụ lệnh sau từ repo root sau khi cấu hình đường dẫn (đầu ra tại một thư mục generated hợp lệ):
   ```bash
   jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb --output 01_EDA.executed.ipynb --output-dir reports
   ```
4. Kiểm tra execution errors, expected sections, manifest coverage, paths, tables/figures mở được, findings khớp thống kê, split disjointness và fingerprint raw CSV trước/sau.
5. Chạy lại hoặc thực hiện kiểm tra tái lập tương đương; so sánh split và thống kê chính, không yêu cầu byte-identical PNG hoặc timestamps.
6. Kiểm tra các edge cases bằng dữ liệu fixture nhỏ riêng biệt nếu cần; không thay fixture cho kết quả EDA trên dataset thật.
7. Báo cáo kết quả từng AC (`PASS`, `FAIL`, `NOT RUN`, `BLOCKED`) và evidence. Nếu môi trường không có CSV, có thể hoàn thành notebook code, kiểm tra cấu trúc và fixture tests; phải ghi rõ chưa chạy dataset thật, không tạo báo cáo có số liệu như thể chạy thành công.

## 10. Implementation priorities

| Priority | Scope | Definition of done |
|---|---|---|
| P0 — Core | Load/schema, structural quality, split protection, descriptive/univariate/target/bivariate, leakage, VIF, exploratory OLS + residuals, findings, export | Chạy end-to-end trên CSV thật và đạt AC liên quan |
| P1 — Complete EDA | Multivariate, feature engineering, interaction/polynomial exploration, model recommendations | Bao phủ toàn bộ applicable analysis ở Section 8/10 và artifact manifest |
| P2 — Optional extension | Formal tests, robust SE, extended sensitivity analysis, supplementary SVG | Chỉ thực hiện nếu phù hợp assumptions và không làm chậm/hỏng core workflow |

P1 là phần của notebook hoàn chỉnh, không phải lý do để bỏ qua. P2 có thể skip có giải thích. Ưu tiên chất lượng phân tích và bằng chứng hơn số lượng biểu đồ.

## 11. Handoff requirements

Khi bàn giao, cần có đường dẫn tới notebook chính, executed notebook nếu có, artifacts, config/cách chạy, data fingerprint, verification results, limitations và unresolved decisions. Không báo hoàn thành thực thi trên dữ liệu thật nếu CSV chưa truy cập được.

Các mặc định trong brief (target Post GPA, split 70/15/15, seed 42, 5 model candidates, 48 table IDs/57 figure IDs, artifact layout, strict holdout, ngôn ngữ trình bày) là quyết định kỹ thuật đề xuất để triển khai nhất quán, không phải tất cả đều là yêu cầu nguyên văn của giảng viên. Chỉ thay đổi khi có lý do, ghi lại quyết định và ảnh hưởng của thay đổi. Số model cuối cùng cần phù hợp số thành viên thật.

Brief này là tài liệu yêu cầu độc lập có thể dùng làm đầu vào cho Spec Kit hoặc triển khai notebook trực tiếp. Trọng tâm là EDA có thể kiểm chứng, không phải tạo thêm specification rườm rà hoặc mở rộng sang training/application ngoài phạm vi.
