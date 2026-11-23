# Data Model: Interaction Linear Regression

## Raw input và target

| Tên | Logical dtype / xử lý | Full | No-Anxiety | No-D |
| --- | --- | --- | --- | --- |
| `Pre_Semester_GPA` | float, P; median/center train-only | ✓ | ✓ | ✓ |
| `Traditional_Study_Hours` | float, S; median/center train-only | ✓ | ✓ | ✓ |
| `Year_of_Study` | categorical, 5 levels nếu đủ | ✓ | ✓ | ✓ |
| `Major_Category` | categorical, 5 levels nếu đủ | ✓ | ✓ | ✓ |
| `Weekly_GenAI_Hours` | float, H; median/center train-only | ✓ | ✓ | ✓ |
| `Primary_Use_Case` | categorical, 5 levels nếu đủ; Ui | ✓ | ✓ | ✓ |
| `Prompt_Engineering_Skill` | categorical, 3 levels nếu đủ; Kj | ✓ | ✓ | ✓ |
| `Tool_Diversity` | numeric, train-only median | ✓ | ✓ | ✓ |
| `Paid_Subscription` | bool/binary, documented 0/1 mapping | ✓ | ✓ | ✓ |
| `Institutional_Policy` | categorical, 3 levels nếu đủ | ✓ | ✓ | ✓ |
| `Perceived_AI_Dependency` | categorical, 10 levels nếu đủ, không continuous | ✓ | ✓ | — |
| `Anxiety_Level_During_Exams` | categorical, 10 levels nếu đủ, không continuous | ✓ | — | — |
| `Post_Semester_GPA` | target float, chỉ training/evaluation; không impute | y | y | y |

`Student_ID` là ID kiểm tra split, không predictor; `GPA_Change`, `Skill_Retention_Score`, `Burnout_Risk_Level`, target và dẫn xuất target không vào X. Raw target một giá trị hữu hạn / hàng; train/validation/test phải giữ nguyên IDs; thiếu target báo count và quy tắc exclusion đã ghi trước, không impute target. Không dùng outcome sau cutoff làm X. Numeric missing được impute train-only cho model; demo raw input bắt buộc đúng mode, sai kiểu/missing lỗi rõ, domain warning không clip.

## Partition và assignment

| Entity | Trường / kiểu | Quy tắc |
| --- | --- | --- |
| Source | `path`, `sha256` string, `schema`, `row_count` integer | SHA-256 của bytes; raw read-only. |
| Assignment | `Student_ID` string, `split` enum train/validation/test; metadata `source_sha256`, `split_id`, `rule`, `seed` | ID unique, coverage đối chiếu source, giữ group trong partition; không tái chia khi đã có. Source hash có thể nằm trong manifest đi kèm. |
| Partition | `split_id`, `partition`, IDs hoặc rows, hash, n | Không giao IDs; 70/15/15 mục tiêu; với source cấp: 35.000/7.500/7.500. |
| CV fold | `fold_id` 0–4, train IDs, holdout IDs, seed/rule/group key | Hai tập fold rời nhau và đều thuộc train; mỗi train row được holdout đúng một lần. |

`data/assignments.csv` có `(Student_ID, split)`; `data/metadata.json` lưu source/assignment/partition hash. `data/splits/{train,val,test}.csv` là các bản sao khớp tại thời điểm audit. Nếu có `frozen_split_ids.csv`, đối chiếu trước khi dùng. Không tái tạo assignment EDA từ hash.

## Encoded design và nested interactions

Đầy đủ: 5 numeric/binary main effects + (5−1)+(5−1)+(5−1)+(3−1)+(3−1)+(10−1)+(10−1) = **39** predictor columns, không tính intercept. Mỗi categorical bỏ một reference; train-only means sau imputation. Tên/cột theo fitted metadata, không hardcode index.

| Model | Terms tích lũy | Shape khi đủ levels |
| --- | --- | --- |
| I0 | main effects | `(n, 39)` |
| I1 | I0 + Hc×K1,K2 | `(n, 41)` |
| I2 | I1 + Hc×U1..U4 | `(n, 45)` |
| I3 | I2 + Ui×Kj, 4×2 | `(n, 53)` |
| I4 | I3 + Hc×Pc + Hc×Sc | `(n, 55)` |

I3 full, no-Anxiety, no-D có shape phụ thuộc levels đã fit; không gán counts full cho ablation. Dữ liệu thiếu level được báo và feature names được lưu; không nhìn sang holdout để bổ sung category. Mọi matrix/prediction phải numeric finite trước fit/score.

## Run và vòng đời

`RunConfig` gồm `run_id`, paths, dataset/split hashes, modes, seed, CV rule, candidate list, metric policy, fit scope và output paths. `FoldResult`: `model`, `fold`, `n_train`, `n_holdout`, `mae`, `mse`, `rmse`, `r2`. `ValidationResult`: model, n, 4 metrics, warnings, fit_scope=train. `ModelLock`: hash config/split, CV scores, tie threshold, selected model/scenario, fit scope, timestamp. `FinalResult`: lock ID, model artifact hash, n, 4 metrics, below-zero/above-four counts và rates, predictions reference.

State machine: `configured → data_validated → development_complete → locked → final_evaluated → exported`. Bất kỳ lỗi integrity chuyển `failed`; chưa chạy ghi `not executed`. Test không được đọc ở bước lựa chọn; model cho validation và final test được fit trên train, không refit trên validation. Replay cùng lock phải nhận diện rõ và không tạo model-selection feedback từ test.
