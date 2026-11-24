# Contract: Model I/O và artifacts

Đây là contract thiết kế, chưa phải interfaces đã implement. Các path cấu hình được; ví dụ dùng `outputs/interaction/<run_id>/`, run ID duy nhất và không ghi đè. Không dựa vào location cá nhân.

## Data input

| Input | Yêu cầu |
| --- | --- |
| Three partitions | CSV train, validation, test, schema nhất quán; cấu hình path; có target và ID nếu có để audit. |
| Single labeled dataset | CSV có `split` là `train`, `validation`, `test` (mapping tên khai báo); không đưa cột split vào X. |
| Raw + assignment | Raw CSV + ID/partition mapping, hash nguồn đúng; join one-to-one và đủ coverage, không nhân dòng. |
| Frozen identity | Chọn assignment nhóm đã xác nhận; kiểm `data/splits/frozen_split_ids.csv`, assignment nhóm/EDA có sẵn, các partitions và provenance. Nếu khác nhau, fail với thông báo so sánh. `data/assignments.csv` hiện là ứng viên, không tự nhận là EDA gốc. |

Missing path/schema/hash mismatch/overlap → error có path/field và hướng sửa, không sửa raw hoặc resplit. Nếu hoàn toàn không có split và chỉ thị create-once đã được đồng bộ constitution: tạo 70/15/15 seed 42, ghi ID/split, source hash/rule/config và chống overwrite; chưa đủ điều kiện thì báo `not executed`. Không có raw/split → báo expected paths.

## Transformer contract

`InteractionFeatures(model_id, scenario, ...)`: sklearn `fit(X, y=None)`, `transform(X)`, `get_feature_names_out(input_features=None)`; cloneable và fitted attributes không thay đổi khi transform. Input dataframe n×12 cho full; n×11 no-Anxiety; n×10 no-D; tên chính xác theo [data-model.md](../data-model.md). Output n×p float64 finite, feature order/names ổn định trong cùng model sau save/reload. `fit` tính train-only medians, category modes, H/P/S means sau numeric impute, OHE categories/reference; không fit bất kỳ thống kê trên transform.

Feature identity dùng mapping origin/level/term do encoder cung cấp; không tách tên giả định theo `_`. `handle_unknown='ignore'`, reference dropped và explicit warning tên feature/value, số lượng; all-zero giống reference, nên không tuyên bố unknown có ý nghĩa bằng reference. Không mutate fitted categories khi gặp unknown; ghi metadata warnings validation/test mà không dùng chúng thay model. Với raw input sai dtype/missing/malformed: `ValueError` có field và sửa như thế nào; unknown category cảnh báo và vẫn dự đoán theo all-zero; numeric ngoài domain hợp lệ cảnh báo, không clip. Với training missing: numeric median và categorical mode học từ fit; all-missing cột lỗi rõ. Values được normalize cùng chính sách ở fit/predict.

## Trained pipeline

| Thành phần | Contract |
| --- | --- |
| Estimator | `LinearRegression(fit_intercept=True)`; OLS/no regularization; pipeline chứa preprocessing + estimator fitted. |
| Model variants | I0–I4 theo [data-model.md](../data-model.md); I3 full/no-Anxiety/no-D được fit riêng. |
| Fit lifecycle | model train-only cho validation và final test sau lock; frozen fitted object không được fit trên validation hoặc refit/mutate để prediction. |
| Identity | `model_version` string, `model_id`, `scenario`, `fit_scope`, expected raw schema, feature_names, reference categories, split hash, code/environment versions, artifact SHA-256. |
| Serialization | `models/<model_id>_<scenario>_<fit_scope>.joblib` và provenance JSON. Chỉ load joblib từ trusted source; validate version/schema/hash trước dùng; lỗi version mismatch phải rõ. |
| Demo | `predict_student(raw_dict, model)` trả mapping có `prediction` float, `model_version`, `model_id`, `scenario`, `warnings` list; invalid input raise `ValueError`. `model` là fitted trusted model kèm metadata, không phải tên model tùy ý chưa resolve. |

No-anxiety chỉ phù hợp pre-exam nếu **mọi input còn lại** đã được xác minh có tại cutoff; no-D cũng tương tự. Nếu timing unknown, chỉ cho phép conditional/retrospective và thông báo. Widget dùng cùng contract fallback; changing AI Hours không là causal intervention.

## Output paths và schemas

| Artifact | Nội dung tối thiểu |
| --- | --- |
| `artifacts/data_manifest.json` | `schema_version`, `source_sha256`, `assignment_sha256`, `split_id`, `partitions` với `path`, `sha256`, `rows`, `id_key`, `overlap_checks`, `group_checks`, `warnings`, `checked_at`. Không ghi raw row IDs. |
| `outputs/interaction/<run_id>/executed_notebook.ipynb` | nbformat 4, saved outputs chỉ từ run đó, execution result/status và config liên kết tới provenance. |
| `tables/cv_folds.csv` | `run_id,model_id,scenario,fold,n_train,n_holdout,mae,mse,rmse,r2`; mỗi model/fold có 1 row. |
| `tables/model_comparison.csv` | `model_id,scenario,n_terms,cv_mae_mean,cv_mae_sd,cv_mse_mean,cv_mse_sd,cv_rmse_mean,cv_rmse_sd,cv_r2_mean,cv_r2_sd,val_mae,val_mse,val_rmse,val_r2`; sd `ddof=1`, validation train-only. |
| `tables/subgroups.csv` | `run_id,model_id,scenario,partition,group_type,group_value,n,rmse,warning`; nhóm n<30 cảnh báo. |
| `tables/predictions_<partition>.csv` | `row_key,y_true,y_pred,residual` với `residual=y_true-y_pred`; row_key là pseudonymous stable key, không chứa Student_ID trực tiếp. Output test chỉ sau lock; tránh public identifiers. |
| `metrics/model_lock.json` | `schema_version,run_id,split_id,split_hash,config_hash,candidates,folds_hash,cv_results,selection_metric,tie_relative=0.001,selected_model,scenario,fit_scope=train,locked_at`. Ghi trước test. |
| `metrics/final_test.json` | `run_id,lock_hash,model_hash,fit_scope,n,mae,mse,rmse,r2,n_below_0,n_above_4,pct_below_0,pct_above_4,status,evaluated_at`; percentages 100×count/n. |
| `metrics/provenance.json` | `schema_version,run_id,code_revision,python_version,dependency_versions,source_sha256,assignment_sha256,partition_hashes,seed,folds_hash,model_id,scenario,fit_scope,feature_names,reference_levels,imputation_values,centering_means,metrics,artifact_hashes,created_at`. Thiếu thông tin ghi `unknown`/lý do. |
| `figures/*.png`, `reports/figures/` | ≥10 mục đích, title/labels/dataset scope; ghi manifest path/hash, lưu bản xuất báo cáo theo constitution. Figure và bảng cùng metric phải cùng run ID, split, partition và model; đối chiếu số điểm, nhãn và giá trị theo tolerance đã công bố trước khi export. |

Model-lock và run provenance immutable sau final test; log replay kiểm tra kết quả với tolerance mà không làm mới lựa chọn. Trước mỗi bước test verify lock/config/model hashes; nếu model/test dùng cho tuning, status `invalid`, không khôi phục holdout cũ bằng đổi tên run.
