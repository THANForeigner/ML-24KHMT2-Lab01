# Specification Quality Checklist: THÂN — Interaction Linear Regression

**Purpose**: Validate specification completeness and quality before proceeding to planning  
**Created**: 2026-11-23  
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No internal framework or class design; named notebook, Python fallback and serialized model are explicit user-facing deliverables.
- [x] Focused on user value and learning needs.
- [x] Written for stakeholders, with statistical terminology required by the exercise.
- [x] All mandatory sections completed.

## Requirement Completeness

- [x] No `[NEEDS CLARIFICATION]` markers remain; open issues are documented as requested.
- [ ] Requirements are fully unambiguous: the shared split and approval of constitution 2.0.0 remain to be confirmed; CV tie threshold, unknown-category policy and subgroup warning threshold are specified.
- [x] Success criteria are measurable, with real-data outcomes conditional on valid inputs.
- [x] Success criteria describe observable outcomes; named notebook/file formats are explicit deliverable constraints.
- [x] Acceptance scenarios cover all nine requested stories.
- [x] Edge cases are identified.
- [x] Scope is clearly bounded.
- [x] Dependencies and assumptions are identified.

## Feature Readiness

- [ ] All functional requirements have fully resolved acceptance criteria; shared split and governance approval remain open.
- [x] User scenarios cover primary flows.
- [ ] Measurable outcomes are not yet demonstrated: this is a draft specification, not an implementation or execution result.
- [x] No internal implementation design leaks into the specification.

## Notes

- Confirm the shared split and actual team approval before finalizing protocol-dependent decisions. Specification drafting and other independent planning are not blocked by absent external documents.
- `REFERENCE_REQUIREMENTS.md` and the THÂN assignment tab were not identified. The Lab 01 PDF was read via the matching Drive file. Constitution 2.0.0 is proposed and retains the I0–I4 formulas.
- Checklist status describes specification quality, not whether the feature or its tests have been implemented.

## Checklist chuyên đề: Lab 01, notebook-first và ML leakage

**Mục đích**: Review chất lượng yêu cầu trước khi phân rã tasks; các mục kiểm tra nội dung đặc tả, không khẳng định implementation đã chạy.  
**Ngày tạo**: 2026-11-23  
**Nguồn**: [spec.md](../spec.md), [plan.md](../plan.md), [constitution 1.1.0](../../../.specify/memory/constitution.md), [Lab 01 PDF](https://drive.google.com/file/d/192i0AvnmsLPdTSn_c1bnl87_2khKc1b7/view).  
**Quyền review**: Các mục mới để `[ ]`; reviewer chỉ đánh `[x]` khi yêu cầu đủ rõ và đã được review. Dấu `[x]` không có nghĩa implementation hoàn thành. `$speckit-implement` không tự thay marker.

### Độ đầy đủ của hợp đồng dữ liệu

- [x] CHK001 Target `Post_Semester_GPA`, đúng 12 raw inputs và sự khác nhau giữa input khi training/inference đã được định nghĩa đầy đủ và nhất quán? [Completeness; Spec: FR-002, FR-003](../spec.md#functional-requirements)
- [x] CHK002 Exclusion list gồm ID, target và các outcome/dẫn xuất gây leak đã rõ ràng, kể cả `GPA_Change` khi có? [Completeness; Spec: FR-003](../spec.md#functional-requirements)
- [x] CHK003 Prediction cutoff và quy tắc availability cho Anxiety, Dependency và các feature khác có timing chưa xác minh đã phân biệt dự báo trước kỳ thi với conditional/retrospective? [Clarity; Spec: FR-003, FR-014](../spec.md#functional-requirements)
- [x] CHK004 Quy tắc xử lý missing target, predictor thiếu, sai kiểu, ngoài domain và category chưa thấy đã được nêu rõ cho training và inference? [Coverage; Spec: Edge Cases, FR-007](../spec.md#edge-cases)

### Định nghĩa mô hình và preprocessing

- [x] CHK005 Công thức lồng nhau I0–I4 và 39/41/45/53/55 predictor terms đã được chỉ rõ cùng điều kiện đủ levels và quy ước không tính intercept? [Clarity; Spec: FR-006, FR-007](../spec.md#functional-requirements)
- [x] CHK006 Reference-coded OHE, reference levels và cách xử lý unknown đều có yêu cầu rõ ràng, kể cả việc all-zero không phân biệt được reference? [Completeness; Spec: FR-007](../spec.md#functional-requirements)
- [x] CHK007 Dependency và Anxiety được định nghĩa là categorical; hierarchy main effects và centering Hc/Pc/Sc trên training partition tương ứng đã được yêu cầu nhất quán? [Consistency; Spec: FR-007, FR-008](../spec.md#functional-requirements)
- [x] CHK008 Phạm vi loại trừ có phân biệt **bình phương/quadratic, spline và hinge bị cấm** với các tích tương tác hai biến được cho phép trong I1–I4? [Clarity; Spec: FR-006, FR-007](../spec.md#functional-requirements)
- [x] CHK009 Yêu cầu thiếu category level trong train hoặc fold đã nêu cách báo số terms thực tế, không dò test để ép đủ số chiều? [Edge Case; Spec: Edge Cases, FR-007](../spec.md#edge-cases)

### Split, CV và bảo vệ holdout

- [x] CHK010 Quy tắc frozen assignment, source hash, ID duy nhất, coverage, partition disjointness và group integrity đã đủ rõ để bảo đảm mọi model dùng cùng split? [Completeness; Spec: FR-004](../spec.md#functional-requirements)
- [x] CHK011 Vai trò train/validation/test, 5-fold CV trên train, group-aware và seed 42 khi phù hợp đã được nêu rõ, không fit preprocessing ngoài train/fold? [Consistency; Spec: FR-008, FR-009](../spec.md#functional-requirements)
- [x] CHK012 Khi assignment khác nhau, split bị lỗi hoặc thiếu một partition, đặc tả có nêu hành vi dừng và không âm thầm tái chia/đổi dòng? [Exception Coverage; Spec: FR-004, Edge Cases](../spec.md#edge-cases)
- [x] CHK013 Chính sách chọn model theo mean CV RMSE, ngưỡng 0,1%, tie-break, model lock và cấm dùng test để tuning đã có thứ tự và điều kiện đo được? [Measurability; Spec: FR-010](../spec.md#functional-requirements)
- [x] CHK014 Cách xử lý test đã bị dùng để tuning và trạng thái `invalid`/holdout mới đã được xác định rõ? [Recovery; Spec: Edge Cases, FR-010](../spec.md#edge-cases)

### Metrics, biểu đồ và diễn giải

- [x] CHK015 MAE/MSE/RMSE/R², từng fold, mean ± sd, validation, final test, tỷ lệ dự đoán ngoài 0–4 và quy tắc không clip đã được yêu cầu đầy đủ? [Completeness; Spec: FR-009–011](../spec.md#functional-requirements)
- [x] CHK016 Yêu cầu so sánh, incremental ablation, no-Anxiety/no-D, subgroup counts/RMSE và diagnostics đã chỉ rõ data scope và conditions so sánh công bằng? [Coverage; Spec: FR-013, FR-018](../spec.md#functional-requirements)
- [x] CHK017 Conditional slopes, joint contrasts, profile cố định, reference, đơn vị và ý nghĩa coefficient đã có yêu cầu giải thích đủ để tái tính? [Clarity; Spec: FR-012](../spec.md#functional-requirements)
- [x] CHK018 Ít nhất 10 mục đích biểu đồ, title/labels/caption, phân vùng dữ liệu và 1–3 câu insight từ số liệu thực có tiêu chí nghiệm thu rõ? [Measurability; Spec: FR-015, SC-003](../spec.md#measurable-outcomes)
- [x] CHK019 Nguồn run, scope, metric và quy tắc đối chiếu giữa bảng và plots cùng loại đã được quy định đủ rõ để tránh hình/bảng mâu thuẫn? [Gap; Spec: FR-015, FR-016, NFR-003](../spec.md#functional-requirements)
- [x] CHK020 Yêu cầu nêu synthetic theo bằng chứng, uncertainty, limitations, association ≠ causation và không mặc định statistical significance đã nhất quán? [Consistency; Spec: FR-005, FR-013, NFR-004](../spec.md#functional-requirements)

### Notebook, demo và bàn giao

- [x] CHK021 Yêu cầu nội dung tiếng Việt, giải thích interaction, ví dụ tính tay, công thức, data dictionary, EDA và thứ tự học trước training đã đủ để review notebook-first? [Completeness; Spec: US1, US2, FR-001](../spec.md#user-story-1--hiểu-bài-toán-và-interaction-priority-p1)
- [x] CHK022 Hợp đồng nhập raw features, prediction, what-if, các mode theo availability và fallback widget đã xác định đủ inputs, outputs, warning và điều kiện sử dụng? [Coverage; Spec: FR-014, US8](../spec.md#functional-requirements)
- [x] CHK023 Notebook validation, clean-kernel Run All, Windows/Kaggle/Colab CPU, dependency, đường dẫn và thiếu data đã được nêu cùng kết quả `not executed` khi chưa chạy? [Nonfunctional; Spec: FR-016, FR-017, NFR-002–003](../spec.md#functional-requirements)
- [x] CHK024 Tests yêu cầu cho encoding, centering, split, no leakage, term counts, inference, serialization và notebook smoke đã bao phủ cả trường hợp lỗi và phân biệt fixture với benchmark thật? [Scenario Coverage; Spec: FR-017](../spec.md#functional-requirements)
- [x] CHK025 Environment, README, executed notebook, figures/tables, model, split/fold identifiers và provenance đã có định nghĩa đủ để tái lập và truy vết? [Completeness; Spec: FR-016, NFR-001](../spec.md#functional-requirements)
- [x] CHK026 Yêu cầu tài liệu/bằng chứng bàn giao phần THÂN đã ánh xạ tới các phần báo cáo Lab 01: motivation, data, models, coefficients, comparison, application và references; phạm vi báo cáo nhóm/PDF/video/ZIP đã phân biệt với deliverable feature? [Completeness; Spec: US9, Assumptions; Lab 01 §3–4](../spec.md#assumptions)

### Xung đột và blocker trước tasks

- [ ] CHK027 Chính sách tạo split khi hoàn toàn chưa có đã được thống nhất giữa spec, plan và constitution, bao gồm việc không ghi đè/tái chia split hiện có? [Conflict; Spec: FR-004, Assumptions; Plan: Constitution Check](../spec.md#assumptions)
- [x] CHK028 Phạm vi I4 đã thống nhất là tùy chọn, được quyết định trước CV; khi chạy phải đúng công thức và 55 predictor terms nếu đủ levels? [Consistency; Spec: FR-006, US3; Plan: Constitution Check](../spec.md#functional-requirements)
- [x] CHK029 Model cuối, preprocessing và validation scores đã thống nhất dùng fit train-only; validation và test được giữ riêng theo chỉ thị gần nhất? [Consistency; Spec: FR-010; Plan: Constitution Check](../spec.md#functional-requirements)
- [x] CHK030 Split được nhóm dùng chung và cách xử lý khác biệt với assignment EDA gốc đã được xác nhận hoặc có quy trình quyết định trước run, không coi hash EDA là assignment hoàn chỉnh? [Dependency; Spec: Assumptions, Open Issues](../spec.md#assumptions)
- [x] CHK031 Nguồn `REFERENCE_REQUIREMENTS.md` và tab THÂN chưa xác định có được gắn nhãn rõ, và có kế hoạch đối chiếu khi sẵn có thay vì giả định đã xác thực? [Dependency; Spec: Assumptions, Open Issues](../spec.md#assumptions)

### Nhận định để chuẩn bị `/speckit.tasks`

- **Trạng thái protocol:** Đã đồng bộ I4 tùy chọn, train-only và create-once trong spec, plan và constitution 2.0.0 đề xuất. CHK027 còn mở vì chưa có bằng chứng nhóm phê duyệt thay đổi constitution; không tự ghi nhận approval.
- **Blocker cho thực nghiệm chung:** CHK030 cần xác nhận assignment chung trước khi chạy/so sánh kết quả nhóm; có thể soạn tasks độc lập cho kiểm tra dữ liệu trong lúc chờ.
- **Nguồn và đối chiếu:** FR-019 quy định đối chiếu bảng/plots; CHK031 ghi rõ nguồn chưa xác định cần được kiểm tra khi có. Nguồn ngoài repo không là tiền điều kiện duy nhất để viết tasks độc lập.
- Checklist chuyên đề này chỉ đánh giá chất lượng yêu cầu. Các mục chưa đánh dấu chưa phải kết luận test hoặc thực nghiệm thất bại.
