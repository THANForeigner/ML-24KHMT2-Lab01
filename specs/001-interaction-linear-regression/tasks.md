---
description: "Dependency-ordered tasks for THÂN Interaction Linear Regression"
---

# Tasks: THÂN Interaction Linear Regression

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md), [data-model.md](data-model.md), [model-io.md](contracts/model-io.md), [quickstart.md](quickstart.md).

**Format**: `- [ ] Tnnn [P] [USn] Công việc — Pre: ...; Output: ...; Verify/evidence: ...; Gate: ...`. `[P]` chỉ dùng cho task không ghi cùng file và không phụ thuộc task chưa xong trong cùng nhóm. Mọi đường dẫn tính từ repository root. Các lệnh là **lệnh nghiệm thu dự kiến sau khi task liên quan được tạo**, không phải khẳng định đã chạy.

**Quy ước chung**: `python` là Python ≥3.11 trong environment đang kích hoạt; `pytest` là `python -m pytest`. Run thật dùng `outputs/interaction/<run_id>/`, không ghi đè run. Tests FIRST: gate đỏ là assertion/import thất bại đúng lý do thiếu tính năng, không chấp nhận lỗi fixture/dependency; sau implementation chạy lại phải xanh. Tests fixture không dùng final holdout; test actual data chỉ đọc **train** để kiểm schema/term count. ID/split audit trên toàn dữ liệu chỉ kiểm toàn vẹn, không tính test metrics. Không tự sửa/xóa dữ liệu hay split có sẵn.

## Phase 0 — Audit và đồng bộ protocol

**Mục tiêu**: phân biệt thông tin đã có với điểm chưa thống nhất trước implementation; không chạy model.

- [ ] T001 Audit `src/lab01_ml/`, `tests/`, `requirements.txt`, `notebooks/` và ghi kết quả vào `outputs/interaction/audit.md` — Pre: repo hiện tại; Output: danh sách module rỗng/tái sử dụng và phiên bản dependencies; Verify/evidence: `rg --files src tests notebooks` và `python -m pip list`; Gate: nêu rõ không có kết quả training mới.
- [ ] T002 Audit `data/assignments.csv`, `data/metadata.json`, `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv` và raw `data/raw/ai_student_impact_dataset (1).csv`, bổ sung hash/ID overlap/schema/counts vào `outputs/interaction/audit.md` — Pre: T001; Output: split candidate và báo sai khác EDA gốc; Verify/evidence: `python -m pytest tests/test_leakage_and_splits.py -k audit_real_split` sau T011; Gate: hashes, assignment, 35k/7.5k/7.5k và IDs khớp hoặc ghi blocker, không overwrite.
- [ ] T003 Đồng bộ quyết định đã được người dùng chỉ định vào `specs/001-interaction-linear-regression/spec.md`, `.specify/memory/constitution.md` và `specs/001-interaction-linear-regression/plan.md`: create-once chỉ khi hoàn toàn chưa có split, I4 tùy chọn, final model fit train-only — Pre: T001; Output: diff governance, rationale, version và trạng thái approval thật; Verify/evidence: `rg -n 'train-only|I4|split' specs/001-interaction-linear-regression/spec.md .specify/memory/constitution.md specs/001-interaction-linear-regression/plan.md`; Gate: không mâu thuẫn; approval governance ghi thực tế, không tự bịa, phần phụ thuộc dừng nếu chưa đạt.
- [ ] T004 Ghi quyết định assignment chung hoặc unresolved vào `outputs/interaction/audit.md`, đối chiếu `data/assignments.csv` với split EDA/nhóm khi có — Pre: T002; Output: split ID/hash, nguồn và bằng chứng xác nhận; Verify/evidence: `rg -n 'split|assignment|approval' outputs/interaction/audit.md`; Gate: thiếu xác nhận không chặn fixture tests nhưng chặn so sánh nhóm/real final evaluation; không tạo split mới để vượt gate.

## Phase 1 — Setup

- [ ] T005 [P] Khai báo trực tiếp Python ≥3.11, joblib, pytest, Jupyter/nbconvert và dependency hiện có trong `requirements.txt`, ipywidgets optional trong `README.md` — Pre: T001; Output: danh sách cài đặt rõ; Verify/evidence: `python -m pip install -r requirements.txt` và `python -m pip check`; Gate: cài được không cần GPU.
- [ ] T006 [P] Tạo `configs/interaction.yaml` với seed 42, paths cấu hình, cutoff, candidates I0–I3 và I4 nếu bật trước CV, tie 0.001 và output root — Pre: T001; Output: config không chứa đường dẫn cá nhân; Verify/evidence: `rg -n '42|0.001|interaction' configs/interaction.yaml`; Gate: configuration không chứa test-derived values.
- [ ] T007 [P] Tạo cấu trúc notebook và output không ghi đè tại `notebooks/interaction_linear_regression.ipynb`, `outputs/interaction/.gitkeep` và hướng dẫn sơ bộ `docs/INTERACTION_QUICKSTART.md` — Pre: T001; Output: notebook hợp lệ nbformat 4 rỗng có tiêu đề, nơi lưu output và quickstart ghi planned; Verify/evidence: `python -c "import nbformat; nbformat.validate(nbformat.read('notebooks/interaction_linear_regression.ipynb',4))"`; Gate: chưa tuyên bố notebook đã execute.

## Phase 2 — Foundational: Tests FIRST

**Mục tiêu**: tạo tests có assertions độc lập trước code; task T008–T013 chỉ có thể song song vì ghi files khác nhau. Test count thực tế chỉ dùng `data/splits/train.csv`, không nhìn validation/test để chọn feature.

- [ ] T008 [P] Viết `tests/test_features.py` với toy numeric assertions cho median→mean centering H/P/S, dummy reference, interaction values, hierarchy, unknown all-zero + warning và category missing — Pre: T005; Output: named tests với expected numbers tính tay; Verify/evidence: `python -m pytest tests/test_features.py -q`; Gate: đỏ đúng vì transformer chưa implement.
- [ ] T009 [P] Viết `tests/test_feature_counts.py` kiểm 39/41/45/53/55 trên fixture đủ levels và `data/splits/train.csv` khi có, xác nhận no-D không có Dependency/Anxiety terms và Dependency/Anxiety categorical — Pre: T005; Output: test fixture + test thực train-only skip có lý do nếu thiếu data; Verify/evidence: `python -m pytest tests/test_feature_counts.py -q`; Gate: đỏ đúng tính năng, không ép counts khi thiếu levels.
- [ ] T010 [P] Viết `tests/test_preprocessing.py` cho train/fold-only median, category vocabulary/means và bất biến khi thay validation/test giả lập — Pre: T005; Output: tests độc lập fixture; Verify/evidence: `python -m pytest tests/test_preprocessing.py -q`; Gate: đỏ đúng tính năng, không ghi chép test thật.
- [ ] T011 [P] Viết `tests/test_leakage_and_splits.py` cho unique IDs, overlap, membership/hash/coverage, missing/invalid split fail không overwrite, leakage exclusions và audit split thật không tính metrics — Pre: T005; Output: unit tests + `test_audit_real_split`; Verify/evidence: `python -m pytest tests/test_leakage_and_splits.py -q`; Gate: fixture đỏ đúng; audit thực chỉ đọc structural data.
- [ ] T012 [P] Viết `tests/test_training.py` cho 5 folds train-only, metrics recompute, tie 0.1%, model lock, final fit train-only và chặn test trước lock — Pre: T005; Output: hand-calculated fixture assertions; Verify/evidence: `python -m pytest tests/test_training.py -q`; Gate: đỏ đúng tính năng, không gọi final test thực.
- [ ] T013 [P] Viết `tests/test_inference.py` và `tests/test_notebook_smoke.py` cho schema/modes/fallback/reload, notebook nbformat và smoke fixture-only — Pre: T005; Output: test dự đoán toy và smoke rõ status; Verify/evidence: `python -m pytest tests/test_inference.py tests/test_notebook_smoke.py -q`; Gate: đỏ đúng phần chưa implement; fixture tách biệt real run.

## Phase 3 — US1: Notebook đọc hiểu bài toán (P1)

**Independent test**: đọc notebook đến ví dụ tay và đối chiếu công thức với `data-model.md`; không cần training.

- [ ] T014 [US1] Viết mục 01–04 trong `notebooks/interaction_linear_regression.ipynb`: mục tiêu, 12 features, target/exclusions, cutoff và roles split — Pre: T007, T013; Output: Markdown Việt + LaTeX, mục đích/input/output; Verify/evidence: `python -m pytest tests/test_notebook_smoke.py -k structure -q`; Gate: notebook hợp lệ, phần mở đầu đủ nội dung.
- [ ] T015 [US1] Viết mục 10–11 trong `notebooks/interaction_linear_regression.ipynb` về OLS, I0–I4, Hc/Pc/Sc, Ui/Kj và ví dụ tính tay slope/contrast — Pre: T014, T008; Output: code cells toy và lời giải, không trình bày fixture là dữ liệu thật; Verify/evidence: `python -m pytest tests/test_features.py -k toy -q` sau T018; Gate: tính tay và code khớp.

## Phase 4 — US2: Dữ liệu, schema, EDA (P1)

**Independent test**: fixture hợp lệ cho load/EDA đến trước training; thiếu file/field báo actionable.

- [ ] T016 [US2] Implement input validation, target/exclusion/availability trong `src/lab01_ml/data/loader.py` và errors path/schema — Pre: T003, T011; Output: API load ba file, một file nhãn split hoặc raw+assignment; Verify/evidence: `python -m pytest tests/test_leakage_and_splits.py -k 'schema or missing or exclusion' -q`; Gate: tests xanh, không tải data.
- [ ] T017 [US2] Viết mục 05–09 trong `notebooks/interaction_linear_regression.ipynb` về schema, data quality, train EDA và split integrity, nêu synthetic theo source và timing — Pre: T014, T016; Output: Markdown/cells data input, EDA trước fit; Verify/evidence: `python -m pytest tests/test_notebook_smoke.py -k data -q`; Gate: fixture chạy đến EDA, không xem test để chọn features.

## Phase 5 — US3: Transformer và I0–I4 (P1)

**Independent test**: fit từng model trên fixture đủ levels, đối chiếu từng giá trị interaction và feature names/counts.

- [ ] T018 [US3] Implement sklearn-compatible `fit/transform/get_feature_names_out` trong `src/lab01_ml/features/engineering.py` với median train, OHE drop-first/ignore warning, mean sau impute và dynamic Ui/Kj — Pre: T008–T010, T016; Output: transformer cloneable và fitted state; Verify/evidence: `python -m pytest tests/test_features.py tests/test_preprocessing.py -q`; Gate: toy, unknown và train-only xanh.
- [ ] T019 [US3] Implement I0–I4 nested feature sets, main effects và no-D/no-Anxiety trong `src/lab01_ml/features/engineering.py` — Pre: T018, T009; Output: counts 39/41/45/53/55 khi đủ levels; Verify/evidence: `python -m pytest tests/test_feature_counts.py -q`; Gate: fixture và actual train nếu có đạt, thiếu levels báo thay vì ép.
- [ ] T020 [US3] Implement OLS pipeline factory `LinearRegression(fit_intercept=True)` trong `src/lab01_ml/models/factory.py` — Pre: T019; Output: 5 pipeline nested và variants, không inverse X'X; Verify/evidence: `python -m pytest tests/test_features.py tests/test_feature_counts.py -q`; Gate: clone/predict và hierarchy đạt, quadratic/spline không có.
- [ ] T021 [US3] Thêm mục 12 vào `notebooks/interaction_linear_regression.ipynb` để kiểm tên features, shape I0–I4 và reference từ train — Pre: T017, T020; Output: cells minh họa từng bước; Verify/evidence: `python -m pytest tests/test_notebook_smoke.py -k features -q`; Gate: counts hiển thị từ fitted train, không hardcode kết quả thật.

## Phase 6 — US4: Frozen split, CV và lựa chọn (P1)

**Independent test**: fixture 5-fold có fold ids, metrics/tie tính tay và selection bất biến khi thay fixture test.

- [ ] T022 [US4] Implement đọc/validate frozen assignment trong `src/lab01_ml/data/splitting.py` và manifest `artifacts/data_manifest.json` — Pre: T003, T011, T016; Output: reusable split identity, conflict fail; Verify/evidence: `python -m pytest tests/test_leakage_and_splits.py -q`; Gate: no overlap/no overwrite, assignment khớp.
- [ ] T023 [US4] Implement create-once chỉ khi hoàn toàn thiếu split trong `src/lab01_ml/data/splitting.py`, output `data/splits/frozen_split_ids.csv` + source hash — Pre: T003 được phê duyệt, T022; Output: 70/15/15 seed 42, group-safe; Verify/evidence: `python -m pytest tests/test_leakage_and_splits.py -k 'create_once or no_overwrite' -q`; Gate: chỉ fixture/test sandbox, không tạo lại split hiện tại; nếu governance chưa cho phép ghi blocked.
- [ ] T024 [US4] Implement 5-fold group-aware/seed 42, clone/fold fit và CV/validation MAE/MSE/RMSE/R² trong `src/lab01_ml/models/training.py` và `src/lab01_ml/models/evaluation.py` — Pre: T012, T020, T022; Output: fold metrics/mean±sample sd, train-only validation; Verify/evidence: `python -m pytest tests/test_training.py -k 'cv or metrics or leakage' -q`; Gate: xanh, không evaluate final test.
- [ ] T025 [US4] Implement selection threshold 0.1%, ít terms, I-index tie và model lock trong `src/lab01_ml/models/training.py` — Pre: T024; Output: `outputs/interaction/<run_id>/metrics/model_lock.json` contract; Verify/evidence: `python -m pytest tests/test_training.py -k 'tie or lock or selection' -q`; Gate: deterministic, lock ghi trước test, validation không thay quyết định.
- [ ] T026 [US4] Viết mục 13–18 trong `notebooks/interaction_linear_regression.ipynb`: CV, bảng validation, selection, ablation I0→I4 — Pre: T021, T024, T025; Output: cells đọc development outputs và giải thích; Verify/evidence: `python -m pytest tests/test_notebook_smoke.py -k development -q`; Gate: fixture notebook không đụng final test thật.

## Phase 7 — US5: Final evaluation contract (P1)

**Independent test**: fixture kiểm lock, fit train-only và một lần score; không sử dụng test project thật trong phase này.

- [ ] T027 [US5] Implement final model fit train-only và chỉ cho phép một final evaluation sau lock có `run_id` trong `src/lab01_ml/models/training.py` và `src/lab01_ml/models/evaluation.py` — Pre: T003, T012, T025; Output: final model và metrics schema, replay cùng lock đánh dấu riêng; Verify/evidence: `python -m pytest tests/test_training.py -k 'final or holdout' -q`; Gate: fixture xác nhận không fit trên validation, test không đọc trước lock và không clip.
- [ ] T028 [US5] Implement actual-vs-predicted, residual-vs-fitted và residual histogram trong `src/lab01_ml/visualization/plots.py` — Pre: T027; Output: functions nhận predictions có dataset scope, không tạo số giả; Verify/evidence: `python -m pytest tests/test_notebook_smoke.py -k residual_plots -q`; Gate: fixture figures có labels, no model selection.
- [ ] T029 [US5] Viết mục 26–27 trong `notebooks/interaction_linear_regression.ipynb` để lock, test một lần bằng model train-only và giải thích bounds/plots — Pre: T026–T028; Output: cells nêu rõ validation và test giữ riêng; Verify/evidence: `python -m pytest tests/test_notebook_smoke.py -k final_gate -q`; Gate: smoke dùng fixture, real test chưa execute.

## Phase 8 — US6: Diễn giải interaction (P1)

**Independent test**: toy profile so sánh conditional slope/joint contrast tính tay với chênh lệch predicted.

- [ ] T030 [P] [US6] Viết tests `tests/test_interpretation.py` cho coefficient units, reference, slope I3/I4, joint contrast và scaler back-transform nếu bật — Pre: T005; Output: toy expected calculations; Verify/evidence: `python -m pytest tests/test_interpretation.py -q`; Gate: đỏ đúng phần chưa implement.
- [ ] T031 [US6] Implement coefficient table, slope, fixed-profile contrast và coefficient back-transform khi có scaler trong `src/lab01_ml/models/evaluation.py` — Pre: T030, T020; Output: dev interpretation tables; Verify/evidence: `python -m pytest tests/test_interpretation.py -q`; Gate: results khớp finite differences và units.
- [ ] T032 [US6] Implement conditional-slope heatmap và adjusted prediction curves theo Use×Skill trong `src/lab01_ml/visualization/plots.py` — Pre: T028, T031; Output: plot functions dùng fixed profile + reference; Verify/evidence: `python -m pytest tests/test_interpretation.py -k plots -q`; Gate: title/labels/profile đúng và association ≠ causation.
- [ ] T033 [US6] Viết mục 19–21 trong `notebooks/interaction_linear_regression.ipynb` về coefficients, slopes, contrasts và curves — Pre: T026, T031, T032; Output: Markdown/cells tiếng Việt, figures PNG; Verify/evidence: `python -m pytest tests/test_notebook_smoke.py -k interpretation -q`; Gate: fixture diễn giải có references, không causal.

## Phase 9 — US8: Input và what-if (P1)

**Independent test**: raw full/no-Anxiety/no-D fixture predict; unknown warning, cutoff error, fallback chạy không widget.

- [ ] T034 [US8] Implement `predict_student(raw_dict, model)` trong `src/lab01_ml/models/inference.py`, trusted model reload và validation full/no-Anxiety/no-D — Pre: T013, T020; Output: prediction/model/scenario/warnings; Verify/evidence: `python -m pytest tests/test_inference.py -q`; Gate: invalid input lỗi rõ, unknown warning, không clip, không load untrusted joblib.
- [ ] T035 [US8] Implement fixed-covariate AI Hours what-if plot trong `src/lab01_ml/visualization/plots.py` — Pre: T032, T034; Output: function nhận raw profile/model, nhãn scenario; Verify/evidence: `python -m pytest tests/test_inference.py -k what_if -q`; Gate: inputs khác giữ nguyên, plot không diễn giải nhân quả.
- [ ] T036 [US8] Viết mục 28–29 trong `notebooks/interaction_linear_regression.ipynb` với widget optional và fallback `predict_student` — Pre: T029, T034, T035; Output: input mới, forecast và what-if; Verify/evidence: `python -m pytest tests/test_notebook_smoke.py -k demo -q`; Gate: fixture demo chạy cả khi không có widget, modes theo cutoff.

## Phase 10 — US9: Reproducibility và handoff (P1)

**Independent test**: quickstart/fixture tạo artifacts + round-trip model mà không cần dữ liệu thật; thực nghiệm thật tách T046.

- [ ] T037 [US9] Implement provenance, unique run ID, versions, seed, hashes, references và artifact manifest trong `src/lab01_ml/utils/reproducibility.py` — Pre: T022, T025; Output: `outputs/interaction/<run_id>/metrics/provenance.json`; Verify/evidence: `python -m pytest tests/test_training.py -k provenance -q`; Gate: lock/split/artifact truy vết và không ghi đè.
- [ ] T038 [US9] Implement trusted-only joblib export/reload, CSV metrics/tables và figures paths trong `src/lab01_ml/models/inference.py` và `src/lab01_ml/models/evaluation.py` — Pre: T027, T034, T037; Output: `outputs/interaction/<run_id>/{models,tables,figures,metrics}/`; Verify/evidence: `python -m pytest tests/test_inference.py -k 'reload or provenance' -q`; Gate: fixture round-trip khớp, không tạo số benchmark giả.
- [ ] T039 [US9] Cập nhật `README.md` với cài/chạy/test Windows/Kaggle/Colab, schema, expected errors, output paths, split, references, link Lab và trusted joblib — Pre: T005, T037, T038; Output: quickstart đồng bộ `specs/001-interaction-linear-regression/quickstart.md`; Verify/evidence: `rg -n 'pytest|nbconvert|Kaggle|Colab|split' README.md`; Gate: lệnh dự kiến chạy được sau T045.
- [ ] T040 [US9] Viết mục 30 của `notebooks/interaction_linear_regression.ipynb` về limitations, references, no causal claim và reproduce — Pre: T036, T039; Output: Markdown kết luận dựa trên số liệu thực hoặc `not executed`; Verify/evidence: `python -m pytest tests/test_notebook_smoke.py -k references -q`; Gate: không khẳng định thực nghiệm chưa chạy.

## Phase 11 — US7: Sensitivity, subgroup và diagnostics (P2)

**Independent test**: fixture subgroup count/RMSE và rank/VIF singular; so sánh ablation trên cùng development split/folds.

- [ ] T041 [P] [US7] Viết `tests/test_diagnostics.py` với RMSE/count nhóm, n<30 warning, VIF infinity/singular, condition/rank và residual/bounds — Pre: T005; Output: toy expected values; Verify/evidence: `python -m pytest tests/test_diagnostics.py -q`; Gate: đỏ đúng phần chưa implement.
- [ ] T042 [US7] Implement subgroup RMSE/counts, residual/bounds, VIF/condition/rank trong `src/lab01_ml/models/evaluation.py` — Pre: T024, T041; Output: development diagnostics + định nghĩa intercept/tolerance; Verify/evidence: `python -m pytest tests/test_diagnostics.py -q`; Gate: metrics tính lại, singular được báo, không dùng test để chọn.
- [ ] T043 [US7] Implement development ablation I3 full/no-Anxiety/no-D trên cùng split/CV trong `src/lab01_ml/models/training.py` — Pre: T019, T024, T042; Output: comparison tables, counts, scenario; Verify/evidence: `python -m pytest tests/test_training.py -k ablation -q`; Gate: development only trừ biến thể được khai báo final trước lock.
- [ ] T044 [US7] Hoàn thiện mục 22–25 trong `notebooks/interaction_linear_regression.ipynb` và `src/lab01_ml/visualization/plots.py`: subgroup, VIF, ablation no-D/no-Anxiety, residual development — Pre: T033, T040–T043; Output: PNG và 1–3 câu insight Việt dựa trên output; Verify/evidence: `python -m pytest tests/test_notebook_smoke.py -k diagnostics -q`; Gate: n<30 warning, không significance/causation chưa chứng minh.

## Phase 12 — Polish và kiểm thử thực thi CPU

- [ ] T045 Hoàn thiện ≥10 charts khác mục đích trong `notebooks/interaction_linear_regression.ipynb` và `src/lab01_ml/visualization/plots.py`: target/AI distributions, AI quantiles, Use×Skill heatmap, CV error bars, val RMSE, ablation, subgroup, slope heatmap, curves, VIF, actual/prediction/residual — Pre: T017, T026, T029, T033, T044; Output: `outputs/interaction/<run_id>/figures/*.png` và `reports/figures/`; Verify/evidence: `python -m pytest tests/test_notebook_smoke.py -k plots -q`; Gate: fixture kiểm đồ thị có scope/title/labels; real run phải có ≥10 PNG và captions/insights thực.
- [ ] T046 Chạy `python -m pytest tests/ -q` và validate `notebooks/interaction_linear_regression.ipynb` bằng `python -c "import nbformat; nbformat.validate(nbformat.read('notebooks/interaction_linear_regression.ipynb',4))"` — Pre: T008–T045, T049–T050; Output: logs trong `outputs/interaction/qa_summary.md`; Verify/evidence: exit code và log; Gate: xanh hoặc ghi failed/not executed có lý do, không claim benchmark từ fixture.
- [ ] T047 Thực hiện **một** CPU clean-kernel run dữ liệu thực `notebooks/interaction_linear_regression.ipynb` bằng `python -m jupyter nbconvert --to notebook --execute notebooks/interaction_linear_regression.ipynb --output executed_notebook.ipynb --output-dir "outputs/interaction/$env:LAB_RUN_ID"` (PowerShell) — Pre: T003 governance approved, T004 split chung xác nhận, T045–T046 xanh, `LAB_RUN_ID` duy nhất, thư mục run mới và data hợp lệ; Output: notebook executed, lock, một final test, CSV/PNG/JSON/joblib; Verify/evidence: log nbconvert và `outputs/interaction/<run_id>/metrics/model_lock.json` trước `final_test.json`; Gate: không tune bằng test, thiếu data/gate ghi `not executed`, không tạo kết quả giả.
- [ ] T048 Kiểm `outputs/interaction/<run_id>/` và ghi `outputs/interaction/qa_summary.md` với code/version, split/lock hashes, 50k counts, >=10 figures, artifacts, fit train-only, test một lần, metrics recompute, unit/nbformat/run status và limitations — Pre: T046, T047 hoặc blocker; Output: QA report kèm paths và passed/failed/not executed; Verify/evidence: `rg -n 'passed|failed|not executed|split|lock' outputs/interaction/qa_summary.md`; Gate: mọi kết luận có bằng chứng, không chạy lại test để chọn model.

## Dependencies & Execution Order

- T001 → T002 → T004; T003 là gate cho implementation phụ thuộc và T023/T047. T005–T007 sau T001, song song vì ghi paths khác.
- Tests T008–T013 tạo trước code; T030 và T041 trước phần riêng US6/US7. Test đỏ có chủ đích, sau đó green.
- Foundation T016/T018/T019/T020/T022 → CV T024/T025 → final evaluation contract T027 → notebook final cells T029 → real CPU T047. Chỉ T047 mở final test thực; T048 chỉ đọc artifacts, không tính lại lựa chọn và không gọi lại estimator trên test.
- T049 theo T024 và T050 theo T045 phải hoàn thành trước T046 dù ID nằm sau phase QA.
- US1: T014–T015; US2: T016–T017; US3: T018–T021; US4: T022–T026; US5: T027–T029; US6: T030–T033; US8: T034–T036; US9: T037–T040; US7: T041–T044. Các story notebook dùng chung một file, vì vậy tích hợp theo thứ tự, không đánh dấu song song.
- Parallel an toàn: T005/T006/T007; T008–T013 sau T005 (khác file); T030 và T041 có thể cùng bắt đầu sau T005 (khác file). Các tác vụ `plots.py`, `evaluation.py`, `training.py`, notebook và `audit.md` dùng chung file nên phải tuần tự hoặc chia commit có điều phối.
- **MVP**: T001–T021 cho bài tự học + toy models (US1–US3), chỉ đánh dấu là phần học tập; chưa phải nghiệm thu dự đoán cuối. Sau đó US4/5/6/8/9, US7, kiểm thử cuối.

## Ma trận truy vết Lab 01 và I0–I4

| Yêu cầu | Task(s) | Output/bằng chứng |
| --- | --- | --- |
| Motivation, continuous target, inputs | T014–T017 | notebook 01–09, data audit |
| Nguồn dữ liệu, preprocessing, visualization | T002, T016–T019, T037, T045 | manifest, notebook, PNG, provenance |
| OLS main approach và giải thích thiết kế | T015, T018–T021 | feature terms, fitted pipelines, toy assertions |
| I0 39 main-effect predictors | T009, T019, T020 | train/fixture feature names/counts |
| I1 41 = I0 + Hc×K1,K2 | T008–T009, T019 | toy values, hierarchy/counts |
| I2 45 = I1 + Hc×U1..U4 | T008–T009, T019 | toy values, hierarchy/counts |
| I3 53 = I2 + Ui×Kj, primary | T009, T019, T031–T033 | counts, conditional slopes/contrasts |
| I4 55 = I3 + Hc×Pc + Hc×Sc | T009, T019, T031 | counts và slope checks |
| Group có ít nhất số model bằng số thành viên | T003, T020, T039 | I0–I4 và scope báo cáo nhóm; kiểm số thành viên được xác nhận, không tự khẳng định nghiệm thu nhóm |
| So sánh metrics/coefficient & assumptions | T024–T026, T030–T033, T041–T045 | CV/val, coefficients, diagnostics và plot captions |
| Regression application nhập dữ liệu & visualize | T034–T036, T045 | `predict_student`, widget/fallback, what-if |
| References, environment, code/data accessibility | T037–T040, T048 | README, provenance, QA; kiểm liên kết trước khi nhóm nộp |
| Final report PDF, demo video, Group ZIP | T039, T048 | Handoff inputs cho nhóm; **deliverables nhóm ngoài feature này**, không đánh dấu đã tạo |
| Reproducibility, no leakage/test tuning | T010–T012, T022–T027, T046–T048 | tests, split/fold/lock hashes, một final evaluation và QA |

## Notes

- Danh sách task là kế hoạch, không phải bằng chứng đã triển khai hoặc đã chạy. Các outputs chưa tạo là `not executed`.
- Không có tác vụ chạy final test thực trước T047 hoặc dùng test để chọn/tune; re-evaluation do lựa chọn mới cần protocol holdout độc lập được phê duyệt.
- Mọi thay đổi protocol và split dùng chung phải ghi review/governance thật; không tự ghi nhận approval của nhóm.

## Phase 13 — Truy vết so sánh và bảng/biểu đồ

- [ ] T049 [US9] Bổ sung contract so sánh outputs của Ridge, Elastic Net, Polynomial và Piecewise khi tương thích target/split/metric/scenario trong `src/lab01_ml/models/evaluation.py` — Pre: T024; Output: bảng so sánh hoặc trạng thái `not executed` khi chưa có outputs; Verify/evidence: `python -m pytest tests/test_training.py -k external_comparison -q`; Gate: không tạo giả kết quả thành viên khác; viết test fixture trước implementation. [FR-018]
- [ ] T050 [US9] Bổ sung đối chiếu run ID, partition, model, labels và metric giữa bảng và plots trong `src/lab01_ml/visualization/plots.py` và `tests/test_notebook_smoke.py` — Pre: T045; Output: kiểm tra bảng/figure cùng run theo tolerance công bố; Verify/evidence: `python -m pytest tests/test_notebook_smoke.py -k figure_table_consistency -q`; Gate: mismatch báo lỗi, test fixture viết trước implementation. [FR-019]
