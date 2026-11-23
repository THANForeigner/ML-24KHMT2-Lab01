# EDA Findings

Dataset SHA-256: `4d911088c4b12d60a450a9acae6b606f4119ebbb48679518e427a4fc00778472`
CSV: `ai_student_impact_dataset (1).csv` · raw rows: 50,000 · columns: 16
Split: train=35,000; validation=7,500; test=7,500; excluded=0.
Scope: detailed EDA on train, prespecified comparisons on validation; test counts/integrity only.
Source: [https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students). Saved metadata: provenance=synthetic, license=CC0-1.0, publisher version=1. Local byte-version match unverified.

## Research questions

| research_question   | finding_id   | status   | evidence_ids                | data_scope                 |
|:--------------------|:-------------|:---------|:----------------------------|:---------------------------|
| RQ01                | FND-001      | resolved | T01 T05 T06 T07 T09 T41     | full_structural_audit      |
| RQ02                | FND-002      | resolved | T15 T29 F16 F49             | validation                 |
| RQ03                | FND-003      | resolved | T19 T20 F24 F32             | train                      |
| RQ04                | FND-004      | resolved | T19 T20 F35                 | train                      |
| RQ05                | FND-005      | resolved | T21 T22 F25 F27 F28         | train                      |
| RQ06                | FND-006      | resolved | T20 T26 F37 F39             | train                      |
| RQ07                | FND-007      | resolved | T24 T25 T26 T33 T35 F24 F53 | train + limited validation |
| RQ08                | FND-008      | resolved | T29 F49                     | validation                 |
| RQ09                | FND-009      | resolved | T27 T28 F41 F54             | train                      |
| RQ10                | FND-010      | resolved | T33 T34 T35 T36 T37 T38     | train + metadata           |
| RQ11                | FND-011      | resolved | T18 T31 T32 F43 F45 F46 F48 | train                      |
| RQ12                | FND-012      | resolved | T40 T42 T48                 | metadata                   |

## Findings

### FND-001 — RQ01

**Observation (full_structural_audit; resolved):** Raw rows=50000, columns=16; raw blank/missing cells=0; exact duplicates removed=0; eligible=50000; logged invalid checks affected rows (summed, may overlap)=0

Audit/cleaning có quy tắc, không impute target.

**Action:** Giữ raw read-only và frozen splits; fit preprocessing train only.

**Limitation:** Completeness không bảo đảm dữ liệu synthetic đại diện thực tế.

**Evidence:** [T01](tables/T01_dataset_overview.csv), [T05](tables/T05_missing_values_summary.csv), [T06](tables/T06_duplicate_records_summary.csv), [T07](tables/T07_invalid_values_summary.csv), [T09](tables/T09_data_quality_report.csv), [T41](tables/T41_split_integrity_report.csv)

### FND-002 — RQ02

**Observation (validation; resolved):** E1 validation n=7500, MAE=0.14127, RMSE=0.182345, R2=0.864381; E0a RMSE=0.49518

Pre GPA là baseline trước kỳ; mức dự đoán được lượng hóa trên holdout validation.

**Action:** Giữ M01 và mean/persistence reference trong training.

**Limitation:** Validation exploratory; chưa phải final test.

**Evidence:** [T15](tables/T15_gpa_change_statistics.csv), [T29](tables/T29_exploratory_regression_results.csv), [F16](figures/F16_pre_vs_post_gpa_with_identity_line.png), [F49](figures/F49_exploratory_model_comparison.png)

### FND-003 — RQ03

**Observation (train; resolved):** Weekly_GenAI_Hours vs Post_Semester_GPA: spearman=0.00794564, valid n=35000; Weekly_GenAI_Hours vs GPA_Change: spearman=0.0313367, valid n=35000

Liên hệ marginal khác với giá trị dự đoán có điều kiện.

**Action:** Kiểm chứng H và quadratic candidate cùng academic controls.

**Limitation:** Timing unknown; GPA Change bị coupling với Pre GPA.

**Evidence:** [T19](tables/T19_pearson_with_target.csv), [T20](tables/T20_spearman_with_target.csv), [F24](figures/F24_ai_hours_vs_gpa_linear_and_binned_trend.png), [F32](figures/F32_gpa_change_by_ai_usage_hours.png)

### FND-004 — RQ04

**Observation (train; resolved):** Traditional_Study_Hours vs Post_Semester_GPA: spearman=0.136739, valid n=35000; Traditional_Study_Hours vs GPA_Change: spearman=0.37143, valid n=35000

Weekly study là candidate trong retrospective academic model.

**Action:** So M02 với M01, xác minh measurement window trước forecast.

**Limitation:** Không mặc định GenAI usage cộng được với study.

**Evidence:** [T19](tables/T19_pearson_with_target.csv), [T20](tables/T20_spearman_with_target.csv), [F35](figures/F35_ai_hours_and_study_heatmap.png)

### FND-005 — RQ05

**Observation (train; resolved):** Major_Category: groups=5, range of group mean=0.0270582, small groups=0; Primary_Use_Case: groups=5, range of group mean=0.102545, small groups=0; Prompt_Engineering_Skill: groups=3, range of group mean=0.0546189, small groups=0

Group differences được báo cùng size và uncertainty.

**Action:** One-hot; flag rare groups; giữ main effects khi kiểm tra interaction.

**Limitation:** Unadjusted differences; multiple comparisons chưa được dùng làm inferential claims.

**Evidence:** [T21](tables/T21_categorical_vs_target_summary.csv), [T22](tables/T22_group_difference_summary.csv), [F25](figures/F25_post_gpa_by_major.png), [F27](figures/F27_post_gpa_by_primary_use_case.png), [F28](figures/F28_post_gpa_by_prompt_skill.png)

### FND-006 — RQ06

**Observation (train; resolved):** Perceived_AI_Dependency vs Post_Semester_GPA: spearman=-0.00939856, valid n=35000; Perceived_AI_Dependency vs GPA_Change: spearman=-0.0118433, valid n=35000; retention Spearman=-0.0544346, n=35000

Dependency là ordinal, timing chưa rõ; retention là auxiliary outcome.

**Action:** Dùng rank để mô tả; one-hot cho conditional model; retention tuyệt đối không vào X.

**Limitation:** Không coi score ordinal là interval scale tự động.

**Evidence:** [T20](tables/T20_spearman_with_target.csv), [T26](tables/T26_subgroup_summary.csv), [F37](figures/F37_dependency_and_anxiety_heatmap.png), [F39](figures/F39_ai_hours_dependency_and_skill_retention.png)

### FND-007 — RQ07

**Observation (train + limited validation; resolved):** Fixed quadratic validation RMSE=0.156321; E3 RMSE=0.157109; delta=-0.000787979; coefficient=-0.000197372; cross-tab cells=130; grouped-trend cells=60

Observed curvature là hypothesis cho specification; interaction grids là descriptive.

**Action:** M04 giữ H khi thêm centered H²; kiểm chứng interaction có lower-order terms.

**Limitation:** Một validation comparison định trước, chưa tuning degree hoặc xác nhận causal moderation.

**Evidence:** [T24](tables/T24_academic_ai_group_summary.csv), [T25](tables/T25_interaction_analysis_summary.csv), [T26](tables/T26_subgroup_summary.csv), [T33](tables/T33_derived_feature_definitions.csv), [T35](tables/T35_original_vs_transformed_comparison.csv), [F24](figures/F24_ai_hours_vs_gpa_linear_and_binned_trend.png), [F53](figures/F53_linear_vs_polynomial_relationship.png)

### FND-008 — RQ08

**Observation (validation; resolved):** E3 minus E2: delta MAE=-0.00278995, delta RMSE=-0.00544708, delta R2=0.00710219; validation n=7500

Negative delta error / positive delta R² means incremental association-based predictive benefit within this scenario.

**Action:** Giữ cùng rows/metrics và ghi rõ M03 là conditional khi timing chưa xác minh.

**Limitation:** Không gọi validation là test; exam anxiety có temporal risk cao.

**Evidence:** [T29](tables/T29_exploratory_regression_results.csv), [F49](figures/F49_exploratory_model_comparison.png)

### FND-009 — RQ09

**Observation (train; resolved):** E3 rank=40/40; encoded terms=39; infinite VIF terms=0; max finite VIF=2.76782; flagged quantitative high-correlation pairs=0

Rank/VIF xét predictor design có reference category và intercept.

**Action:** Không tự động xóa theo heuristic; dùng Ridge candidate nếu coefficients thiếu ổn định.

**Limitation:** Term-level VIF phụ thuộc encoding/feature set; raw condition number phụ thuộc đơn vị.

**Evidence:** [T27](tables/T27_highly_correlated_predictor_pairs.csv), [T28](tables/T28_vif_analysis.csv), [F41](figures/F41_vif_comparison.png), [F54](figures/F54_correlation_before_after_engineering.png)

### FND-010 — RQ10

**Observation (train + metadata; resolved):** 16 observed features have encoding plans; 6 derived train candidates; hours additivity verified=False

Impute/one-hot train-only; transformations dựa domain và hierarchy.

**Action:** Median numeric, mode + one-hot categorical; scale polynomial/Ridge trong train pipeline.

**Limitation:** No forced total/ratio; no equal-spacing assumption for ordinal numeric interactions.

**Evidence:** [T33](tables/T33_derived_feature_definitions.csv), [T34](tables/T34_derived_feature_statistics.csv), [T35](tables/T35_original_vs_transformed_comparison.csv), [T36](tables/T36_encoding_recommendations.csv), [T37](tables/T37_scaling_recommendations.csv), [T38](tables/T38_feature_selection_recommendations.csv)

### FND-011 — RQ11

**Observation (train; resolved):** Train residual skew=-0.218421; abs standardized residual >3: 188; Cook D >4/n: 1641; leverage >2p/n: 1387; n=35000, p=40

Residual diagnostics có thể chỉ ra giới hạn linear mean, variance và bounded target.

**Action:** Inspect residual structure; cân nhắc HC3, polynomial candidate hoặc regularization; không tự xóa flags.

**Limitation:** Independence không xác minh; normality cần cho inference, không phải yêu cầu predictors chuẩn.

**Evidence:** [T18](tables/T18_gpa_boundary_concentration.csv), [T31](tables/T31_regression_assumption_diagnostics.csv), [T32](tables/T32_influential_observations_summary.csv), [F43](figures/F43_residual_vs_fitted.png), [F45](figures/F45_qq_plot.png), [F46](figures/F46_scale_location.png), [F48](figures/F48_residuals_vs_key_predictors.png)

### FND-012 — RQ12

**Observation (metadata; resolved):** Publisher provenance=synthetic; license=CC0-1.0; version=1; verified forecast features=1

Synthetic patterns không tự khái quát đến sinh viên thật.

**Action:** Cần cohort validation và measurement timestamps trước triển khai.

**Limitation:** Generator, representativeness, true GPA floor/precision, course PDFs và team size chưa xác minh.

**Evidence:** [T40](tables/T40_feature_availability_matrix.csv), [T42](tables/T42_dataset_bias_and_limitations.csv), [T48](tables/T48_dataset_limitations.csv)

## Model candidates

| model   | method                                   | features                                                                                                                                                                                                                                               | scenario                                                | rationale                                                                                                | evidence_ids    | status                                                              |
|:--------|:-----------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|:----------------|:--------------------------------------------------------------------|
| M01     | Simple Linear Regression                 | Pre_Semester_GPA                                                                                                                                                                                                                                       | pre-semester baseline                                   | E1 vs E0a/E0b in T29                                                                                     | T15 T29         | training candidate; no final test evaluation                        |
| M02     | Multiple Linear Regression Academic      | Pre_Semester_GPA; Major_Category; Year_of_Study; Traditional_Study_Hours                                                                                                                                                                               | conditional / retrospective unless all timings verified | Academic increment measured E2 vs E1; scenario differs                                                   | T19 T21 T29 T40 | training candidate                                                  |
| M03     | Multiple Linear Regression Academic + AI | Pre_Semester_GPA; Major_Category; Year_of_Study; Traditional_Study_Hours; Weekly_GenAI_Hours; Primary_Use_Case; Prompt_Engineering_Skill; Tool_Diversity; Paid_Subscription; Perceived_AI_Dependency; Institutional_Policy; Anxiety_Level_During_Exams | conditional / retrospective; anxiety high temporal risk | E3 minus E2: delta MAE=-0.00278995, delta RMSE=-0.00544708, delta R2=0.00710219; validation n=7500       | T28 T29 T40     | training candidate; verify timing or exclude anxiety for deployment |
| M04     | Polynomial Linear Regression             | M03 + centered H^2; selected reference-coded interactions only with main effects                                                                                                                                                                       | conditional / retrospective                             | Fixed quadratic validation RMSE=0.156321; E3 RMSE=0.157109; delta=-0.000787979; coefficient=-0.000197372 | T24 T28 T33 T35 | training candidate; degree/interaction selection via train CV       |
| M05     | Ridge Regression                         | same eligible encoded feature set as chosen M02/M03 scenario; scale within train pipeline                                                                                                                                                              | match selected feature set and availability             | test coefficient stability against OLS based on rank/VIF; confirm course allowance                       | T28 T37 T40     | proposal only; no tuning in this EDA                                |

MAE, MSE, RMSE, R² use the same target/splits/population. Final model count must be at least team size (unknown).

## Unresolved and limitations

| issue                      | status                                           | evidence                                                          | consequence                                                              | action                                                         |
|:---------------------------|:-------------------------------------------------|:------------------------------------------------------------------|:-------------------------------------------------------------------------|:---------------------------------------------------------------|
| synthetic provenance       | synthetic                                        | official user_specified_sources in saved metadata                 | Learns simulated patterns; external validity to real students unverified | No causal or population claims                                 |
| license and version        | CC0-1.0; publisher version=1                     | https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students | Local bytes not authenticated to publisher checksum                      | Record local SHA-256; retain source metadata snapshot          |
| sampling / independence    | unknown                                          | publisher does not describe sampling or generation mechanism      | SE/CI are descriptive; representative inference unsupported              | Audit real cohort before external use                          |
| feature timing             | mostly unknown                                   | T40; source definitions                                           | E2/E3 not verified forecasts                                             | Obtain measurement dates / freeze cutoff                       |
| hours coverage             | study additivity unverified                      | AI usage hours not explicitly restricted to study                 | Total/ratio may misstate study time                                      | Skip unless documented                                         |
| GPA boundaries / precision | upper documented; lower and precision unverified | T18; source approximate ranges                                    | Floor effect unknown; equality tolerance only numerical                  | No inferred floor; no silent prediction clipping               |
| group imbalance            | 0 rare category rows in frequency table          | T12; T21; T24                                                     | Small subgroups have unstable summaries                                  | Flag n<30; report counts                                       |
| holdout policy             | test untouched for analysis                      | T41; split assignments                                            | Validation is exploratory, not final estimate                            | Final test only after model specification frozen               |
| course source PDFs         | unavailable in workspace                         | brief.md references two PDF files                                 | Exact original rubric not verified                                       | Confirm final model count versus team size and Ridge allowance |

No causal AI impact claim; no population generalization from synthetic data; no final test evaluation.