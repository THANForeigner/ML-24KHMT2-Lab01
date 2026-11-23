# Implementation Plan: THÂN Interaction Linear Regression

**Branch**: `001-interaction-linear-regression` (tên do Spec Kit resolve; không xác nhận đã tạo Git branch) | **Date**: 2026-11-23 | **Spec**: [spec.md](spec.md)

**Input**: Spec và constitution 1.1.0; chỉ thị triển khai mới nhất của người dùng.

## Summary

Xây notebook tự học tiếng Việt, minh họa từng bước bằng code cells nhỏ gọi các hàm đã kiểm thử trong package `lab01_ml`. Một transformer sklearn fit imputation, reference OHE, centering và nested interactions trong từng fold; pipeline OLS so sánh I0–I3 và I4 nếu bật trên cùng split/CV. Chọn model bằng CV RMSE theo tie policy, khóa quyết định rồi đánh giá final test bằng model fit train-only; I3 là trọng tâm diễn giải. Dữ liệu và artifacts có provenance; notebook demo nhận raw inputs với fallback Python.

## Technical Context

**Language/Version**: Python ≥3.11; notebook nbformat 4.

**Primary Dependencies**: NumPy, pandas, scikit-learn, matplotlib, seaborn, joblib, nbformat, nbclient/nbconvert, pytest; ipywidgets optional. Dùng dependency hiện có, bổ sung trực tiếp joblib, pytest, nbconvert và Jupyter khi triển khai. Không bắt buộc ipywidgets.

**Storage**: CSV/JSON/PNG/joblib, local filesystem; `outputs/interaction/<run_id>/` cho run outputs, `artifacts/data_manifest.json` và `reports/figures/` theo constitution; chỉ ghi manifest tổng hợp, không sao chép ID không cần thiết vào output công khai.

**Testing**: pytest, nbformat validation, nbclient/nbconvert clean-kernel smoke bằng fixture test-only, full real run riêng khi có dữ liệu; không báo smoke là benchmark.

**Target Platform**: Windows, Kaggle, Colab; CPU, input local/mounted, không tự tải dataset. Root tìm theo `pyproject.toml` và `src/lab01_ml`, cho phép config `LAB_PROJECT_ROOT` khi cần; không dùng đường dẫn máy cá nhân.

**Project Type**: Python package + notebook giáo dục + interface dự đoán tại notebook.

**Performance Goals**: Hoàn thành 50.000 bản ghi và 5-fold trên CPU; log wall time và peak memory khi chạy; không áp đặt SLA hoặc RMSE chưa đo.

**Constraints**: Không GPU/server/network runtime; 5 folds, fold-only fit, test một lần sau lock; model cuối fit train-only; joblib chỉ load từ nguồn tin cậy. Không sửa dữ liệu raw và không ghi đè run khác.

**Scale/Scope**: 50.000 rows (35k/7,5k/7,5k nếu source được xác nhận); 12 features, 39–55 encoded predictors với đủ levels; 5 models + 2 sensitivity; ≥30 phần notebook, ≥10 plots thực.

## Constitution Check

*GATE pre-research and post-design*: Các xung đột protocol đã được đồng bộ theo constitution 2.0.0 đề xuất và chỉ thị gần nhất; phê duyệt nhóm cho constitution và split chung vẫn chưa có bằng chứng. Không tuyên bố các phê duyệt này đã hoàn tất.

| Kiểm tra | Trạng thái / bằng chứng và xử lý |
| --- | --- |
| Academic integrity, notebook-first, citations, association ≠ causation | Thiết kế phù hợp; kiểm tra output thật và review khi triển khai. |
| Pre-existing common split, no resplit | Constitution 2.0.0 đề xuất cho phép create-once nếu hoàn toàn thiếu split; repo đã có `data/assignments.csv` và ba partitions nên reuse. Nhóm chưa xác nhận split; không báo đã phê duyệt. |
| I0–I4 / scope | I0–I3 bắt buộc; I4 là tùy chọn, có thể bật trước CV; kiểm counts cả năm trên fixture/train đủ levels. |
| Selection / holdout | Model cuối fit train-only theo chỉ thị mới nhất; validation giữ riêng, model lock trước test. |
| Preprocessing / leakage / timing | Fit mỗi fold, giữ test kín; development no-D/no-anxiety; mode phù hợp cutoff; không fit trên validation. |
| Review / outputs | Đảm bảo manifest `artifacts/data_manifest.json`, báo cáo figures theo constitution; mọi tests/thực nghiệm đều `not executed` cho đến khi chạy. |

## Project Structure

### Documentation (this feature)

```text
specs/001-interaction-linear-regression/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/model-io.md
└── tasks.md                 # sau /speckit.tasks; không tạo trong bước plan
```

### Source Code (repository root)

```text
notebooks/interaction_linear_regression.ipynb
src/lab01_ml/
├── data/{loader,splitting,preprocessing}.py
├── features/engineering.py
├── models/{factory,training,evaluation,inference}.py
├── visualization/plots.py
└── utils/{paths,reproducibility}.py
tests/{test_features,test_leakage_and_splits,test_notebook_smoke}.py
data/{raw/,splits/,assignments.csv,metadata.json}
outputs/interaction/<run_id>/{executed_notebook.ipynb,tables/,figures/,models/,metrics/}
artifacts/data_manifest.json
reports/figures/
requirements.txt
README.md
```

**Structure Decision**: Các module package, configs và tests hiện tại chỉ là stub; không có `src/interaction_features.py`, `train_interaction.py`, `evaluate_interaction.py`, `predict_interaction.py`. Mở rộng package sẵn có, không tạo bản sao song song. Nếu có hàm EDA reusable trong `scripts/build_notebook.py`, trích có chọn lọc sau audit; không import builder có side effect hoặc thay notebook của thành viên khác. `data/splits/frozen_split_ids.csv` chưa có: `data/assignments.csv` là assignment ứng viên hiện có, không ghi đè; tên mới chỉ dùng nếu tạo assignment mới được phép hoặc export không thay membership và bảo toàn provenance.

## Architecture and Workflow

1. **Load/validate**: cấu hình đọc một file có `split`, ba partitions, hoặc raw + assignment. Kiểm hash raw/partition/assignment, schema, IDs, overlap, coverage, duplicates và group leakage. Nếu assignment khả dụng mâu thuẫn, dừng; không âm thầm chọn. Tạo `artifacts/data_manifest.json`; thiếu data báo đường dẫn cần cung cấp.
2. **Frozen split**: ưu tiên assignment nhóm đã xác nhận. `data/assignments.csv` khớp partitions và raw hash theo audit 2026-11-23; metadata cho biết khác EDA gốc. Không resplit hay thay assignment có lỗi. Constitution 2.0.0 đề xuất create-once nếu hoàn toàn thiếu split, dùng `random_state=42`, 70/15/15, lưu Student_ID/split/source SHA-256/rule; duplicate/group-safe và ghi counts, không dùng y để tạo tuning signal.
3. **Transformer**: sklearn `fit/transform/get_feature_names_out`, constructor cloneable; validate 12 raw columns và exclude target/ID. Numeric median fit từ train partition; sau imputation tính mean H/P/S trên cùng partition; categorical mode fit từ train, ties theo thứ tự ổn định, nếu toàn missing fail rõ; Dependency/Anxiety chuyển categorical theo normalization ổn định. `OneHotEncoder(drop='first', handle_unknown='ignore')` fit trên partition, ghi `categories_`, `drop_idx_`, tên cột; tìm Ui/Kj bằng metadata feature, không parse nhãn tùy tiện. Cảnh báo unseen và all-zero tương đương reference. Khi fold thiếu level ghi actual feature names/counts và không dùng test để ép chiều.
4. **Nested models**: I0 `X_main`; I1 + `Hc×K1..K2`; I2 + `Hc×U1..U4`; I3 + 8 `Ui×Kj`; I4 + `Hc×Pc`, `Hc×Sc`. 39/41/45/53/55 predictor terms, chưa tính intercept, chỉ khi đủ train levels; giữ main effects. OLS `LinearRegression(fit_intercept=True)` với dense design vừa đủ, sklearn least squares, không đảo `XᵀX`. Không scale mặc định; nếu tùy chọn scale, fit từng fold và back-transform cả coefficients, interactions và intercept; kiểm slope/contrast bằng dự đoán profile gốc.
5. **Development**: 5-fold train với group-aware CV nếu group IDs hợp lệ, nếu không KFold shuffle seed 42; lưu fold assignment/seed, fold RMSE/MAE/MSE/R² và mean ± sample sd (ddof=1). Mỗi fold clone pipeline. Fit riêng model train-only cho validation; I3 full/no-anxiety/no-D đánh giá chỉ development trừ khi cấu hình đó được chọn trước test; main candidates là I0–I3 và I4 nếu bật trước CV.
6. **Lock & evaluate**: chọn mean CV RMSE thấp nhất; tương đương nếu ≤1,001×min (min=0 chỉ đúng 0), chọn ít terms nhất rồi I-index nhỏ hơn. Validation mô tả/sanity check, không đảo quyết định. Lưu model-lock gồm split/config hash, candidates, CV scores, model/scenario, policy, fit_scope=train và timestamp trước khi mở test. Dùng model fit trên train để transform test đúng một lần theo execution record. Metric/diagnostic test không dùng để đổi model hay policy; rerun để tái lập cùng lock phải được đánh dấu là replay, không phải cơ hội chọn model mới.
7. **Interpret/demo**: I3 slopes `β_H+γ_k+δ_u`, I4 thêm `θPc+φSc`; joint contrast bao gồm main và Ui×Kj. Báo references, units, support và giả định. No-Anxiety mode cho pre-exam nếu các feature khác sẵn có; no-D cho cutoff không có cả D; full gắn conditional/retrospective khi timing chưa rõ. UI widget optional và `predict_student(raw_dict, model)` luôn có.
8. **Provenance & diagnostics**: version/schema, dataset/assignment hashes, references, imputation/means, fold definitions, feature names, versions, code revision, fit scope, selection/lock and artifact hashes vào JSON. Rank/SVD của design nêu rõ intercept/tolerance; condition number, VIF từ train design, cảnh báo constant/collinear/undefined. Dùng định nghĩa thống nhất residual `y−ŷ` và không clip predictions khi tính metrics.

### Notebook outline (30 phần, mỗi phần có Markdown và code/output tương ứng)

01 mục tiêu; 02 dictionary; 03 leakage; 04 cấu hình; 05 load/schema; 06 EDA target train; 07 EDA AI hours train; 08 Use×Skill descriptive train; 09 split integrity/counts; 10 OLS + interaction LaTeX; 11 OHE/centering ví dụ tay; 12 I0–I4 feature names/shapes; 13 giải thích CV; 14 train/CV/validation; 15 bảng models; 16 CV error bars; 17 validation RMSE; 18 incremental ablation; 19 coefficients; 20 conditional slopes; 21 adjusted curves; 22 subgroup RMSE/counts; 23 VIF/rank/condition; 24 no-Anxiety/no-D; 25 residual diagnostics development; 26 model lock rồi test bằng model train-only; 27 final test plots; 28 predict raw row; 29 widget what-if + fallback; 30 limitations/references/reproduce. Mỗi plot title/labels/scope và 1–3 câu dựa trên output thực; ít nhất 10 mục đích khác nhau.

### Test strategy matrix

| Tests dự kiến | Requirement / bằng chứng |
| --- | --- |
| `test_features.py` | FR-002/003/006–008: 39/41/45/53/55 full-level fixture, hierarchy, feature names động, medians/means, OHE drop-first, unknown warning, categorical D, clone/serialization và scaler back-transform nếu bật. |
| `test_leakage_and_splits.py` | FR-004/008–010/013: raw/assignment hash, counts, overlap, reuse/no overwrite, conflicting IDs, one-time creation sau governance; group/fold disjointness, thay validation/test không đổi fitted state, tie boundaries/model-lock/train-only và test chỉ sau lock. |
| `test_inference.py` (mở rộng stub) | FR-014/016: modes/cutoff, raw schema/missing/unseen/domain, `predict_student`, no clipping, model reload trusted source, manifest/model compatibility. |
| `test_training.py` (mở rộng stub) | FR-009–013: 5 folds, recompute 4 metrics/mean/sd, I3 separately, ablation, subgroup counts, slopes/contrasts so với dự đoán trực tiếp, rank/VIF/zero variance handling. |
| `test_notebook_smoke.py` | FR-001/005/015–017: nbformat 4 validation, 30 headings, kernel sạch fixture-only, output path và missing-data actionable; real run riêng xác nhận ≥10 figures/thuyết minh và executed artifact. |

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
| --- | --- | --- |
| Create-once là thay đổi so với constitution 1.1.0 | Người dùng muốn hỗ trợ raw-only; bản 2.0.0 đề xuất cho phép tạo một lần khi hoàn toàn chưa có split. | Repo có split nên tái sử dụng; không ghi đè hoặc thay split lỗi. |
| Phê duyệt constitution và split chung chưa có bằng chứng | Cần phân biệt bản đề xuất với phê duyệt thật. | Không tự ghi nhận quyết định hoặc thay split của nhóm. |
