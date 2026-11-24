"""Build the educational Elastic Net Regression notebook; no model execution occurs here."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
cells = []


def section(title, explanation, code=None):
    cells.append(nbf.v4.new_markdown_cell('## ' + title + '\n\n' + explanation))
    if code:
        cells.append(nbf.v4.new_code_cell(code))


section('01. Mục tiêu & Động cơ',
        'Xây dựng mô hình **Elastic Net Regression** để dự đoán `Post_Semester_GPA` (phân tích có điều kiện / What-if Analysis) '
        'trên bộ dữ liệu synthetic *AI Impact on Students*. Khi mở rộng không gian đặc trưng chứa nhiều tương tác và đa thức, '
        'hiện tượng đa cộng tuyến (multicollinearity) trở nên nghiêm trọng. Elastic Net kết hợp ưu điểm co phương sai của Ridge ($L_2$) '
        'và năng lực chọn biến thưa của Lasso ($L_1$), khắc phục nhược điểm của Lasso khi các biến có tương quan cao (grouping effect). '
        'Đây là dữ liệu synthetic; các kết luận chỉ mang tính tương quan có điều kiện, không suy diễn quan hệ nhân quả.')

section('02. Thiết lập Môi trường & Thư mục Output',
        'Tự động xác định thư mục gốc của repository, thiết lập các thư mục đầu ra trong `outputs/elastic_net/` '
        'và nạp các thư viện cần thiết.',
        '''from pathlib import Path
import os, sys, json, joblib, warnings
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'pyproject.toml').exists() and (p/'src/lab01_ml').exists()), None)
if ROOT is None:
    raise FileNotFoundError('Không tìm thấy project root; vui lòng mở notebook trong cấu trúc repo Lab 01.')
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from lab01_ml.data.loader import load_frozen, TARGET, EXCLUDED_COLUMNS
from lab01_ml.features.elastic_net_features import ElasticNetFeatures, INPUTS, H, P, S, NUMERIC, CATEGORICAL
from lab01_ml.models.factory import create_elastic_net_pipeline, create_ols_pipeline
from lab01_ml.models.elastic_net_training import (
    tune_elastic_net_cv, select_best_elastic_net, fit_final_elastic_net,
    train_elastic_net_ablation, metrics
)
from lab01_ml.models.elastic_net_evaluation import (
    extract_sparsity_metrics, calculate_selection_frequency,
    check_hierarchy, check_gpa_bounds, evaluate_on_validation
)
from lab01_ml.models.inference import predict_student
from lab01_ml.visualization.plots import (
    plot_gpa_distribution, plot_ai_hours_distribution, plot_ai_hours_vs_gpa_change,
    plot_group_heatmap, plot_hyperparameter_surface, plot_regularization_path,
    plot_cv_rmse_comparison, plot_validation_comparison, plot_coefficient_magnitudes,
    plot_selection_frequency, plot_residuals, plot_what_if_curves
)

OUT = ROOT / 'outputs/elastic_net'
FIG = OUT / 'figures'
TABLES = OUT / 'tables'
MODELS = OUT / 'models'
for p in (FIG, TABLES, MODELS):
    p.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid')
print('Project root:', ROOT)
print('Outputs directory:', OUT)
''')

section('03. Hợp đồng Dữ liệu Đầu vào (12 Raw Features)',
        'Đầu vào của mô hình gồm đúng 12 đặc trưng thô thuộc 4 nhóm (A: Academic, B: AI Usage, C: Contextual, D: Psychological). '
        'Biến mục tiêu duy nhất là `Post_Semester_GPA`.',
        '''input_df = pd.DataFrame({'Tên biến': INPUTS, 'Nhóm': [
    'A (Academic)' if c in [P, S, 'Year_of_Study', 'Major_Category'] else
    'B (AI Usage)' if c in [H, 'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Tool_Diversity', 'Paid_Subscription'] else
    'C (Contextual)' if c == 'Institutional_Policy' else 'D (Psychological)'
    for c in INPUTS
]})
display(input_df)
print('Target:', TARGET)
''')

section('04. Tránh Rò rỉ Dữ liệu (Leakage Guard & Exclusion List)',
        'Các cột `Student_ID`, `GPA_Change`, `Skill_Retention_Score`, `Burnout_Risk_Level` tuyệt đối không được đưa vào $X$. '
        '`GPA_Change` là biến dẫn xuất từ target ($GPA_{post} - GPA_{pre}$), nếu đưa vào sẽ gây rò rỉ mục tiêu hoàn toàn. '
        'Các biến outcome đo lường sau kỳ học cũng không sẵn có tại cutoff dự đoán.',
        '''print('Các cột bị loại trừ bắt buộc:', EXCLUDED_COLUMNS)
assert not any(c in INPUTS for c in EXCLUDED_COLUMNS), 'Lỗi rò rỉ: cột cấm nằm trong INPUTS!'
print('Xác thực danh sách loại trừ: HOÀN TOÀN AN TOÀN.')
''')

section('05. Phân tích Thời điểm Đo lường (Timing & Retrospective)',
        'Do thời điểm thu thập dữ liệu của các biến nhóm D (`Anxiety_Level_During_Exams`, `Perceived_AI_Dependency`) '
        'chưa được chứng thực là trước hay sau kỳ thi, mô hình mang tính hồi cứu (retrospective). '
        'Trước kỳ thi, mức độ lo lắng trong phòng thi chưa tồn tại, do đó cần có kịch bản Group Ablation loại bỏ biến này.')

section('06. Kiểm tra Split Đóng băng & Mã băm (70/15/15)',
        'Nạp 3 tập dữ liệu đã chia sẵn (35.000 train, 7.500 validation, 7.500 test). '
        'Kiểm tra mã băm SHA-256 đối chiếu với `data/metadata.json` và `data/assignments.csv`. '
        'Tập `test` được bảo vệ: chỉ nạp `Student_ID` để audit, không nạp target.',
        '''frames, provenance = load_frozen(ROOT)
train = frames['train']
val = frames['validation']
display(pd.DataFrame([provenance['counts']]))
(OUT / 'train_split_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf8')
print('Mã băm assignment:', provenance['assignment_sha256'])
print('Trạng thái phê duyệt nhóm:', provenance['group_approval'])
''')

section('07. Sơ đồ Dữ liệu & Kiểm tra Dữ liệu Thiếu',
        'Kiểm tra kiểu dữ liệu và số lượng giá trị thiếu (missing values) trên tập train. '
        'Các thông số EDA và tiền xử lý tuyệt đối không tham chiếu sang validation hay test.',
        '''schema_df = pd.DataFrame({
    'Kiểu dữ liệu': train.dtypes,
    'Số lượng thiếu (NaN)': train.isna().sum(),
    'Tỷ lệ thiếu (%)': (train.isna().sum() / len(train)) * 100
})
display(schema_df)
''')

section('08. Phân phối GPA Cuối kỳ (Train EDA)',
        'Khảo sát phân phối của nhãn `Post_Semester_GPA` trên tập train.',
        '''fig, ax = plot_gpa_distribution(train, save_path=FIG/'01_gpa_distribution.png')
plt.show()
print(f"Trung vị GPA train = {train[TARGET].median():.3f}, Mean = {train[TARGET].mean():.3f}, SD = {train[TARGET].std():.3f}")
''')

section('09. Phân phối Giờ dùng AI (Train EDA)',
        'Khảo sát thời gian sử dụng GenAI mỗi tuần (`Weekly_GenAI_Hours`) trên tập train.',
        '''fig, ax = plot_ai_hours_distribution(train, save_path=FIG/'02_ai_hours.png')
plt.show()
print(f"Trung vị Giờ AI = {train[H].median():.2f} giờ, Q1-Q3 = {train[H].quantile(.25):.2f} - {train[H].quantile(.75):.2f} giờ")
''')

section('10. Mối liên hệ Thô: Giờ AI và GPA Change (Train EDA)',
        'Biểu đồ phân tán giữa số giờ AI và thay đổi GPA thuần túy nhằm mục đích mô tả cấu trúc mẫu dữ liệu.',
        '''fig, ax = plot_ai_hours_vs_gpa_change(train, save_path=FIG/'03_ai_gpa_change.png')
plt.show()
''')

section('11. Bảng Phân bố Nhóm Use Case × Prompt Skill (Train EDA)',
        'GPA trung bình quan sát tại các ô kết hợp giữa mục đích sử dụng và kỹ năng prompt.',
        '''fig, ax = plot_group_heatmap(train, save_path=FIG/'04_group_heatmap.png')
plt.show()
''')

section('12. Lý thuyết Toán học Elastic Net & Hàm Mất mát (L1 + L2)',
        'Elastic Net giải bài toán tối ưu lồi kết hợp hai chuẩn phạt $L_1$ và $L_2$ trên các hệ số hồi quy: '
        '$$\\min_{\\beta_0, \\beta} \\frac{1}{2n} \\|y - \\beta_0 \\mathbf{1} - X\\beta\\|_2^2 + \\alpha \\left( l_1\\_ratio \\|\\beta\\|_1 + \\frac{1 - l_1\\_ratio}{2} \\|\\beta\\|_2^2 \\right)$$ '
        '- Khi $l_1\\_ratio = 1$: mô hình trở thành Lasso thuần túy ($L_1$), sinh nghiệm thưa. '
        '- Khi $l_1\\_ratio = 0$: mô hình trở thành Ridge thuần túy ($L_2$), co hệ số đều nhau. '
        '- Khi $0 < l_1\\_ratio < 1$: Elastic Net vừa co cụm các biến tương quan (grouping effect) vừa triệt tiêu các biến nhiễu về đúng 0.')

section('13. Chuẩn hóa Thang đo (StandardScaler) cho Regularization',
        'Do hàm phạt Elastic Net phạt trực tiếp độ lớn $|\beta_j|$ và $\beta_j^2$, các biến có thang đo tự nhiên lớn '
        '(như $Weekly\\_GenAI\\_Hours \\in [0, 40]$) sẽ có hệ số nhỏ và bị phạt ít hơn so với biến dummy $0/1$. '
        'Vì vậy, **bắt buộc phải chuẩn hóa (`StandardScaler`)** toàn bộ ma trận đặc trưng về mean 0, variance 1. '
        'Scaler phải được fit khép kín trong từng fold Cross-Validation để chống rò rỉ dữ liệu.')

section('14. Kỹ thuật Trừ tâm (Centering) & Quy tắc Tránh Đa cộng tuyến hoàn hảo',
        'Đặt $H_c = H - \\overline{H}_{train}$, $P_c = P - \\overline{P}_{train}$, $S_c = S - \\overline{S}_{train}$. '
        'Khi bổ sung thành phần phi tuyến $H_c^2$, **tuyệt đối không giữ đồng thời $H$ thô và $H_c$** khi mô hình có hệ số chặn (intercept) '
        'vì $H = H_c + \\overline{H}_{train} \\cdot \\mathbf{1}$, dẫn đến đa cộng tuyến hoàn hảo (ma trận suy biến).')

section('15. One-Hot Encoding drop-first & Reference Levels',
        'Mỗi biến phân loại bỏ mức đầu tiên làm mức tham chiếu (reference level). '
        'Các danh mục lạ chưa thấy trong quá trình kiểm thử sẽ được mã hóa an toàn thành vector toàn 0 kèm warning.')

section('16. Ba Cấu hình Đặc trưng: EN1, EN2, EN3',
        'Kiểm tra số chiều ma trận thiết kế theo hợp đồng: '
        '- **EN1**: 39 predictor terms (Main effects sau OHE). '
        '- **EN2**: 40 predictor terms (EN1 + $H_c^2$; không giữ raw $H$). '
        '- **EN3**: 54 predictor terms (EN2 + 14 terms tương tác 2 chiều: $H_c \\times U_i$, $H_c \\times K_j$, $U_i \\times K_j$). '
        '- **EN3_no_anxiety**: 45 predictor terms (Ablation loại bỏ `Anxiety_Level_During_Exams`).',
        '''for m, expected in [('EN1', 39), ('EN2', 40), ('EN3', 54)]:
    ft = ElasticNetFeatures(model_type=m).fit(train[INPUTS])
    actual = len(ft.get_feature_names_out())
    print(f'{m}: {actual} terms (kỳ vọng: {expected})')
    assert actual == expected, f'Sai lệch số terms tại {m}!'

ft_ablation = ElasticNetFeatures(model_type='EN3', scenario='no_anxiety').fit(train[INPUTS])
print(f'EN3_no_anxiety: {len(ft_ablation.get_feature_names_out())} terms (kỳ vọng: 45)')
assert len(ft_ablation.get_feature_names_out()) == 45
print('Toàn bộ cấu hình đặc trưng: CHÍNH XÁC 100%.')
''')

section('17. Minh họa Tính toán Biến đổi Đặc trưng trên Một hàng',
        'Đối chiếu phép tính tay giá trị $H_c$ với kết quả xuất ra từ `ElasticNetFeatures`.',
        '''sample_row = train[INPUTS].iloc[[0]]
ft3 = ElasticNetFeatures(model_type='EN3').fit(train[INPUTS])
trans = ft3.transform(sample_row)
print('Tính tay Hc:', float(sample_row[H].iloc[0]) - ft3.means_[H])
print('Output Hc từ Transformer:', trans[0, 0])
feature_df = pd.DataFrame({'Tên đặc trưng': ft3.get_feature_names_out(), 'Giá trị hàng mẫu': trans[0, :]})
display(feature_df.head(15))
''')

section('18. Thiết lập 5-Fold Cross-Validation trên Train',
        'Chia tập train (35.000 dòng) thành 5 folds ngẫu nhiên cố định (`seed=42`). '
        'Mỗi fold fit lại toàn bộ preprocessing và `StandardScaler`.')

section('19. Tinh chỉnh Siêu tham số (Grid Search trên alpha & l1_ratio)',
        'Thực hiện tìm kiếm lưới trên không gian siêu tham số cho cấu hình trọng tâm `EN3`: '
        '$\\alpha \\in [10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0]$ và $l_1\\_ratio \\in [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]$.',
        '''cv_results_en3, fold_details_en3 = tune_elastic_net_cv(
    train, model_type='EN3', scenario='full', n_splits=5, random_state=42
)
cv_results_en3.to_csv(TABLES / 'cv_tuning_grid.csv', index=False)
best_en3 = select_best_elastic_net(cv_results_en3, tie_threshold=0.001)
print(f"Cặp siêu tham số tối ưu chọn theo tie-break: alpha = {best_en3['alpha']}, l1_ratio = {best_en3['l1_ratio']}")
print(f"CV RMSE nhỏ nhất = {best_en3['cv_rmse_mean']:.4f} ± {best_en3['cv_rmse_std']:.4f}, Số hệ số != 0 = {best_en3['mean_non_zero_terms']:.1f}/54")
display(cv_results_en3.sort_values(by='cv_rmse_mean').head(10))
''')

section('20. Bề mặt Sai số Siêu tham số (Heatmap CV RMSE)',
        'Trực quan hóa bề mặt sai số Cross-Validation theo cặp siêu tham số $(\\alpha, l_1\\_ratio)$.',
        '''fig, ax = plot_hyperparameter_surface(cv_results_en3, save_path=FIG/'05_hyperparameter_heatmap.png')
plt.show()
''')

section('21. Quỹ đạo Hệ số hồi quy (Regularization Path)',
        'Khảo sát quá trình co cụm và triệt tiêu các hệ số về 0 khi cường độ phạt $\\alpha$ tăng dần từ $10^{-4}$ đến $10^1$.',
        '''fig, ax = plot_regularization_path(train[INPUTS].iloc[:5000], train[TARGET].iloc[:5000], l1_ratio=best_en3['l1_ratio'], save_path=FIG/'06_regularization_path.png')
plt.show()
''')

section('22. So sánh Sai số CV (CV RMSE ± SD)',
        'So sánh sai số 5-fold CV giữa EN1, EN2, EN3 và baseline OLS I3.',
        '''cv_results_en1, _ = tune_elastic_net_cv(train, model_type='EN1', scenario='full', alphas=[best_en3['alpha']], l1_ratios=[best_en3['l1_ratio']], n_splits=5)
cv_results_en2, _ = tune_elastic_net_cv(train, model_type='EN2', scenario='full', alphas=[best_en3['alpha']], l1_ratios=[best_en3['l1_ratio']], n_splits=5)

model_cv_summary = pd.DataFrame([
    {'model_type': 'EN1', 'cv_rmse_mean': cv_results_en1['cv_rmse_mean'].iloc[0], 'cv_rmse_std': cv_results_en1['cv_rmse_std'].iloc[0]},
    {'model_type': 'EN2', 'cv_rmse_mean': cv_results_en2['cv_rmse_mean'].iloc[0], 'cv_rmse_std': cv_results_en2['cv_rmse_std'].iloc[0]},
    {'model_type': 'EN3', 'cv_rmse_mean': best_en3['cv_rmse_mean'], 'cv_rmse_std': best_en3['cv_rmse_std']}
])
fig, ax = plot_cv_rmse_comparison(model_cv_summary, save_path=FIG/'07_cv_rmse_comparison.png')
plt.show()
display(model_cv_summary)
''')

section('23. Đánh giá trên Tập Validation & So sánh OLS Baseline',
        'Huấn luyện các mô hình trên toàn bộ tập train (35.000 dòng) và đánh giá trên tập validation độc lập (7.500 dòng), '
        'so sánh định lượng với baseline OLS.',
        '''# Fit final models on 100% training data
pipe_en1 = fit_final_elastic_net(train, model_type='EN1', alpha=best_en3['alpha'], l1_ratio=best_en3['l1_ratio'])
pipe_en2 = fit_final_elastic_net(train, model_type='EN2', alpha=best_en3['alpha'], l1_ratio=best_en3['l1_ratio'])
pipe_en3 = fit_final_elastic_net(train, model_type='EN3', alpha=best_en3['alpha'], l1_ratio=best_en3['l1_ratio'])

# Baseline OLS models
pipe_ols_i0 = create_ols_pipeline(model=0).fit(train[INPUTS], train[TARGET])
pipe_ols_i3 = create_ols_pipeline(model=3).fit(train[INPUTS], train[TARGET])

models_to_eval = {
    'OLS_I0': pipe_ols_i0,
    'OLS_I3': pipe_ols_i3,
    'EN1': pipe_en1,
    'EN2': pipe_en2,
    'EN3': pipe_en3
}

val_comparison_df = evaluate_on_validation(models_to_eval, val)
val_comparison_df.to_csv(TABLES / 'model_comparison.csv', index=False)
display(val_comparison_df)

fig, _ = plot_validation_comparison(val_comparison_df, save_path=FIG/'08_validation_rmse_r2.png')
plt.show()
''')

section('24. Phân tích Group Ablation (Loại bỏ Anxiety_Level_During_Exams)',
        'Huấn luyện mô hình `EN3_no_anxiety` để đo lường mức độ suy giảm hiệu năng khi chuyển sang kịch bản dự đoán trước kỳ thi.',
        '''ablation_cv_results, pipe_ablation = train_elastic_net_ablation(
    train, alphas=[best_en3['alpha']], l1_ratios=[best_en3['l1_ratio']], n_splits=5
)
ablation_val_df = evaluate_on_validation({'EN3_Full': pipe_en3, 'EN3_No_Anxiety': pipe_ablation}, val)
ablation_val_df.to_csv(TABLES / 'ablation_comparison.csv', index=False)
display(ablation_val_df)
print(f"Mức tăng RMSE khi bỏ Anxiety: {ablation_val_df.loc[1, 'val_rmse'] - ablation_val_df.loc[0, 'val_rmse']:.4f}")
''')

section('25. Trích xuất Độ thưa (Sparsity) & Độ lớn Hệ số Tối ưu',
        'Trích xuất các hệ số khác 0 của mô hình `EN3` và trực quan hóa độ lớn của các hệ số có ảnh hưởng lớn nhất.',
        '''k_nz, sparsity_pct, coef_df = extract_sparsity_metrics(pipe_en3)
coef_df.to_csv(TABLES / 'non_zero_coefficients.csv', index=False)
print(f"Tổng số terms: {len(coef_df)}")
print(f"Số hệ số khác 0 (|beta| > 1e-6): {k_nz}")
print(f"Tỷ lệ thưa (Sparsity Rate): {sparsity_pct:.2f}%")

fig, ax = plot_coefficient_magnitudes(coef_df, top_n=20, save_path=FIG/'09_coefficient_magnitudes.png')
plt.show()
display(coef_df[coef_df['is_non_zero']].head(15))
''')

section('26. Tần suất Chọn Biến qua 5 Folds Cross-Validation',
        'Thống kê tỷ lệ các folds trong 5-fold CV mà mỗi đặc trưng được giữ lại (hệ số khác 0) '
        'để xác định nhóm biến cốt lõi (core stable features).',
        '''# Lấy fold_coefs của mô hình tối ưu
opt_row = cv_results_en3[(cv_results_en3['alpha'] == best_en3['alpha']) & (cv_results_en3['l1_ratio'] == best_en3['l1_ratio'])].iloc[0]
feature_names = pipe_en3.named_steps['features'].get_feature_names_out()
freq_df = calculate_selection_frequency(feature_names, opt_row['fold_coefs'])
freq_df.to_csv(TABLES / 'selection_frequency.csv', index=False)

fig, ax = plot_selection_frequency(freq_df, top_n=25, save_path=FIG/'10_selection_frequency.png')
plt.show()
print(f"Số lượng biến cốt lõi (chọn trong 5/5 folds): {(freq_df['frequency'] == 1.0).sum()}")
display(freq_df.head(15))
''')

section('27. Kiểm tra Tính Phân cấp Cấu trúc (Hierarchy Check)',
        'Kiểm tra xem Elastic Net có giữ lại biến tương tác khi biến thành phần bị triệt tiêu về 0 hay không.',
        '''coefs_en3 = pipe_en3.named_steps['regressor'].coef_
violations = check_hierarchy(feature_names, coefs_en3)
hierarchy_df = pd.DataFrame(violations)
hierarchy_df.to_csv(TABLES / 'hierarchy_check.csv', index=False)

if len(violations) > 0:
    print(f"Phát hiện {len(violations)} trường hợp vi phạm tính phân cấp cấu trúc:")
    display(hierarchy_df)
    print("Nhận xét: Elastic Net tiêu chuẩn áp dụng phạt term-level riêng lẻ nên không tự động bảo toàn phân cấp.")
else:
    print("Không phát hiện vi phạm phân cấp: Mọi biến tương tác được giữ đều có main effects đi kèm.")
''')

section('28. Kiểm soát Trần Điểm GPA (GPA Ceiling / Floor Check - Không Clip)',
        'Thống kê số lượng và tỷ lệ phần trăm dự đoán trên tập validation vượt trần GPA (> 4.0) hoặc dưới đáy (< 0.0). '
        'Tuyệt đối không tự ý cắt/clip âm thầm để đảm bảo tính trung thực định lượng.',
        '''y_val_preds = pipe_en3.predict(val[INPUTS])
bounds_summary = check_gpa_bounds(y_val_preds)
print('Tổng số mẫu validation:', bounds_summary['total_predictions'])
print('Số dự đoán dưới 0.0:', bounds_summary['below_zero_count'])
print('Số dự đoán vượt trần 4.0:', bounds_summary['above_four_count'])
print(f"Tỷ lệ ngoài miền: {bounds_summary['out_of_bounds_pct']:.2f}% (Không áp dụng hàm clip)")
''')

section('29. Chẩn đoán Phần dư trên Tập Validation (Residual Diagnostics)',
        'Kiểm tra giả định về tính đồng nhất phương sai (homoscedasticity) và phân phối chuẩn của phần dư.',
        '''fig, _ = plot_residuals(val[TARGET], y_val_preds, save_path=FIG/'11_residuals_vs_fitted.png')
plt.show()
''')

section('30. Khóa Mô hình & Bảo vệ Tập Holdout (Model Lock)',
        'Lưu cấu hình mô hình tối ưu vào `model_lock.json` và xuất pipeline hoàn chỉnh ra file `.joblib`.',
        '''lock_info = {
    'selected_model': 'EN3',
    'alpha': float(best_en3['alpha']),
    'l1_ratio': float(best_en3['l1_ratio']),
    'total_terms': len(feature_names),
    'non_zero_terms': k_nz,
    'sparsity_pct': sparsity_pct,
    'cv_rmse_mean': float(best_en3['cv_rmse_mean']),
    'val_rmse': float(val_comparison_df.loc[val_comparison_df['model_name'] == 'EN3', 'val_rmse'].iloc[0]),
    'split_hash': provenance['assignment_sha256'],
    'status': 'locked'
}
(OUT / 'model_lock.json').write_text(json.dumps(lock_info, indent=2), encoding='utf8')
joblib.dump(pipe_en3, MODELS / 'selected_elastic_net_pipeline.joblib')
print('Mô hình tối ưu đã được khóa và lưu tại:', MODELS / 'selected_elastic_net_pipeline.joblib')
''')

section('31. Đánh giá Final Holdout (Chỉ mở khi có phê duyệt chung)',
        'Theo giao thức bảo vệ holdout, tập test chỉ được đánh giá một lần duy nhất khi biến môi trường '
        '`LAB_GROUP_SPLIT_APPROVED = "yes"` được kích hoạt.',
        '''metrics_summary = {
    'selected_model': lock_info,
    'validation_metrics': {
        'rmse': float(val_comparison_df.loc[val_comparison_df['model_name'] == 'EN3', 'val_rmse'].iloc[0]),
        'mae': float(val_comparison_df.loc[val_comparison_df['model_name'] == 'EN3', 'val_mae'].iloc[0]),
        'r2': float(val_comparison_df.loc[val_comparison_df['model_name'] == 'EN3', 'val_r2'].iloc[0]),
        'out_of_bounds_pct': bounds_summary['out_of_bounds_pct']
    },
    'final_test': {'status': 'not executed', 'reason': 'group split approval missing'}
}

if os.environ.get('LAB_GROUP_SPLIT_APPROVED') == 'yes':
    test_frames, _ = load_frozen(ROOT, include_test_target=True)
    test_df = test_frames['test']
    test_preds = pipe_en3.predict(test_df[INPUTS])
    test_scores = metrics(test_df[TARGET], test_preds)
    test_bounds = check_gpa_bounds(test_preds)
    metrics_summary['final_test'] = {
        'status': 'executed',
        **test_scores,
        **test_bounds
    }
    print('Final Test Metrics:', test_scores)
else:
    print('Final test: not executed — split chung chưa được xác nhận bởi nhóm.')

(OUT / 'metrics.json').write_text(json.dumps(metrics_summary, indent=2), encoding='utf8')
''')

section('32. Giao diện Dự đoán Đơn lẻ (Raw Profile Inference với predict_student)',
        'Thử nghiệm hàm `predict_student` nhận dictionary đúng 12 trường raw inputs và trả kết quả hợp lệ.',
        '''sample_student = train[INPUTS].iloc[0].to_dict()
inference_result = predict_student(sample_student, pipe_en3)
print('Dự đoán cho sinh viên mẫu:')
for k, v in inference_result.items():
    print(f"  {k}: {v}")
''')

section('33. Phân tích Kịch bản What-if & Giao diện Tương tác',
        'Mô phỏng sự thay đổi của GPA dự đoán khi quét số giờ sử dụng GenAI từ 0 đến 40 giờ '
        'trên các nhóm mục đích sử dụng và trình độ prompt khác nhau.',
        '''fig, ax = plot_what_if_curves(pipe_en3, sample_student, save_path=FIG/'12_what_if_response_curves.png')
plt.show()
''')

section('34. Giới hạn Nghiên cứu, Đạo đức & Hướng dẫn Tái lập',
        '- **Dữ liệu Synthetic**: Bộ dữ liệu là nhân tạo; không suy luận quan hệ nhân quả. '
        '- **Kịch bản Retrospective**: Thời điểm thu thập dữ liệu trong học kỳ chưa được xác thực độc lập. '
        '- **Giới hạn Lọc Biến Elastic Net**: Elastic Net tiêu chuẩn thực hiện phạt trên từng cột dummy riêng lẻ (term-level), '
        'không tương đương với việc chọn/loại bỏ toàn bộ biến danh mục (group selection). '
        '- **Hướng dẫn Tái lập**: Chạy lại từ kernel sạch (`Restart Kernel & Run All Cells`) hoặc thực thi script '
        '`python scripts/execute_elastic_net_notebook.py`.')

nb.cells = cells
target_path = Path(__file__).resolve().parents[1] / 'notebooks/elastic_net_regression.ipynb'
target_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, target_path)
print(f'Wrote {target_path} with {len(cells)} cells.')
