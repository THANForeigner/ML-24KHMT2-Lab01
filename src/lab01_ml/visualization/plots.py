"""Plotting utilities for Elastic Net and exploratory data analysis."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from lab01_ml.features.elastic_net_features import H, P, TARGET


def plot_gpa_distribution(train_df, save_path=None):
    """Plot distribution of Post_Semester_GPA on training set."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(train_df[TARGET], bins=35, kde=True, ax=ax, color='steelblue')
    ax.set_title('Phân phối GPA cuối kỳ (Post_Semester_GPA) — Tập Train', fontsize=12)
    ax.set_xlabel('GPA cuối kỳ', fontsize=10)
    ax.set_ylabel('Số lượng sinh viên', fontsize=10)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_ai_hours_distribution(train_df, save_path=None):
    """Plot distribution of Weekly_GenAI_Hours on training set."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(train_df[H], bins=35, kde=True, ax=ax, color='darkorange')
    ax.set_title('Phân phối thời gian dùng AI mỗi tuần (Weekly_GenAI_Hours) — Tập Train', fontsize=12)
    ax.set_xlabel('Số giờ AI / tuần', fontsize=10)
    ax.set_ylabel('Số lượng sinh viên', fontsize=10)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_ai_hours_vs_gpa_change(train_df, save_path=None):
    """Scatter plot of AI Hours vs GPA Change (Post - Pre) for exploratory analysis."""
    fig, ax = plt.subplots(figsize=(8, 4))
    gpa_change = train_df[TARGET] - train_df[P]
    sample_idx = train_df.sample(min(2500, len(train_df)), random_state=42).index
    ax.scatter(train_df.loc[sample_idx, H], gpa_change.loc[sample_idx], alpha=0.25, s=12, color='teal')
    ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Baseline: Không đổi GPA')
    ax.set_title('Mối liên hệ giữa Giờ dùng AI và Thay đổi GPA (Mô tả — Tập Train)', fontsize=12)
    ax.set_xlabel('Số giờ AI / tuần', fontsize=10)
    ax.set_ylabel('Thay đổi GPA (Post − Pre)', fontsize=10)
    ax.legend()
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_group_heatmap(train_df, save_path=None):
    """Heatmap of mean GPA across Primary_Use_Case and Prompt_Engineering_Skill."""
    group = train_df.pivot_table(
        index='Primary_Use_Case',
        columns='Prompt_Engineering_Skill',
        values=TARGET,
        aggfunc='mean'
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(group, annot=True, fmt='.3f', cmap='Blues', ax=ax, cbar_kws={'label': 'GPA trung bình'})
    ax.set_title('GPA trung bình quan sát theo Use Case × Prompt Skill — Tập Train', fontsize=12)
    ax.set_xlabel('Kỹ năng Prompt (Prompt Engineering Skill)', fontsize=10)
    ax.set_ylabel('Mục đích sử dụng chính (Primary Use Case)', fontsize=10)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_hyperparameter_surface(cv_results_df, save_path=None):
    """Heatmap of CV RMSE across alphas and l1_ratios."""
    pivot = cv_results_df.pivot(index='alpha', columns='l1_ratio', values='cv_rmse_mean')
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis_r', ax=ax, cbar_kws={'label': 'CV RMSE'})
    ax.set_title('Bề mặt sai số 5-Fold CV RMSE theo cặp siêu tham số (alpha, l1_ratio)', fontsize=12)
    ax.set_xlabel('Tỷ trọng phạt L1 (l1_ratio)', fontsize=10)
    ax.set_ylabel('Cường độ phạt tổng thể (alpha)', fontsize=10)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_regularization_path(X_train, y_train, alphas=None, l1_ratio=0.7, save_path=None):
    """Plot the regularization path showing shrinkage of coefficients as alpha increases."""
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler
    from lab01_ml.features.elastic_net_features import ElasticNetFeatures

    if alphas is None:
        alphas = np.logspace(-4, 1, 30)

    feat = ElasticNetFeatures(model_type='EN3').fit(X_train)
    X_trans = feat.transform(X_train)
    scaler = StandardScaler().fit(X_trans)
    X_scaled = scaler.transform(X_trans)

    coef_paths = []
    for a in alphas:
        en = ElasticNet(alpha=a, l1_ratio=l1_ratio, max_iter=5000, tol=1e-3, random_state=42)
        en.fit(X_scaled, y_train)
        coef_paths.append(en.coef_)

    coef_paths = np.array(coef_paths)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(coef_paths.shape[1]):
        ax.plot(alphas, coef_paths[:, i], alpha=0.6, linewidth=1)
    ax.set_xscale('log')
    ax.set_title(f'Quỹ đạo hệ số hồi quy (Regularization Path) khi alpha tăng (l1_ratio={l1_ratio})', fontsize=12)
    ax.set_xlabel('Hệ số phạt alpha (log scale)', fontsize=10)
    ax.set_ylabel('Độ lớn hệ số chuẩn hóa (Standardized Coefficients)', fontsize=10)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_cv_rmse_comparison(comparison_df, save_path=None):
    """Bar plot comparing CV RMSE across models with error bars."""
    fig, ax = plt.subplots(figsize=(8, 4))
    models = comparison_df['model_type'] if 'model_type' in comparison_df else comparison_df['model_name']
    rmse = comparison_df['cv_rmse_mean']
    std = comparison_df['cv_rmse_std'] if 'cv_rmse_std' in comparison_df else None

    ax.bar(models, rmse, yerr=std, capsize=4, color='cornflowerblue', edgecolor='navy', alpha=0.85)
    ax.set_title('So sánh 5-Fold CV RMSE giữa các cấu hình đặc trưng', fontsize=12)
    ax.set_xlabel('Cấu hình mô hình', fontsize=10)
    ax.set_ylabel('Mean CV RMSE ± SD', fontsize=10)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_validation_comparison(val_comparison_df, save_path=None):
    """Side-by-side bar plots comparing Validation RMSE and R2."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    names = val_comparison_df['model_name']

    ax1.bar(names, val_comparison_df['val_rmse'], color='salmon', edgecolor='firebrick', alpha=0.85)
    ax1.set_title('Validation RMSE (Càng thấp càng tốt)', fontsize=11)
    ax1.set_ylabel('RMSE', fontsize=10)
    ax1.tick_params(axis='x', rotation=30)

    ax2.bar(names, val_comparison_df['val_r2'], color='mediumseagreen', edgecolor='darkgreen', alpha=0.85)
    ax2.set_title('Validation R² (Càng cao càng tốt)', fontsize=11)
    ax2.set_ylabel('R²', fontsize=10)
    ax2.tick_params(axis='x', rotation=30)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, (ax1, ax2)


def plot_coefficient_magnitudes(coef_df, top_n=20, save_path=None):
    """Horizontal bar plot showing magnitudes of top retained non-zero coefficients."""
    non_zero = coef_df[coef_df['is_non_zero']].head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['navy' if c > 0 else 'crimson' for c in non_zero['coefficient'][::-1]]
    ax.barh(non_zero['feature'][::-1], non_zero['coefficient'][::-1], color=colors, alpha=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'Top {len(non_zero)} hệ số khác 0 có tác động lớn nhất (Elastic Net EN3)', fontsize=12)
    ax.set_xlabel('Giá trị hệ số chuẩn hóa (Standardized Beta)', fontsize=10)
    ax.set_ylabel('Tên đặc trưng', fontsize=10)
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_selection_frequency(freq_df, top_n=25, save_path=None):
    """Bar plot showing selection frequency across folds."""
    top_freq = freq_df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_freq['feature'][::-1], top_freq['frequency'][::-1] * 100.0, color='teal', alpha=0.8)
    ax.set_xlim(0, 105)
    ax.axvline(100, color='red', linestyle='--', linewidth=1, label='Cốt lõi (100% các fold)')
    ax.set_title(f'Tần suất được chọn qua 5 Folds Cross-Validation (Top {len(top_freq)} biến)', fontsize=12)
    ax.set_xlabel('Tần suất chọn (%)', fontsize=10)
    ax.set_ylabel('Đặc trưng', fontsize=10)
    ax.legend(loc='lower right')
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax


def plot_residuals(y_true, y_pred, save_path=None):
    """Plot Residuals vs Fitted values to diagnose homoscedasticity."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.scatter(y_pred, residuals, alpha=0.2, s=10, color='purple')
    ax1.axhline(0, color='red', linestyle='--', linewidth=1)
    ax1.set_title('Phần dư vs Giá trị dự đoán (Residuals vs Fitted) — Validation', fontsize=11)
    ax1.set_xlabel('Giá trị GPA dự đoán (Fitted values)', fontsize=10)
    ax1.set_ylabel('Phần dư (Residual = Actual − Pred)', fontsize=10)

    sns.histplot(residuals, bins=35, kde=True, ax=ax2, color='purple')
    ax2.set_title('Phân phối phần dư (Residual Distribution)', fontsize=11)
    ax2.set_xlabel('Phần dư', fontsize=10)
    ax2.set_ylabel('Tần số', fontsize=10)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, (ax1, ax2)


def plot_what_if_curves(pipeline, profile, ai_hours_grid=None, save_path=None):
    """Plot what-if curves varying Weekly_GenAI_Hours across different Use Cases and Prompt Skills."""
    if ai_hours_grid is None:
        ai_hours_grid = np.linspace(0, 40, 50)

    use_cases = ['Summarization', 'Coding Assistance', 'Exam Preparation']
    prompt_skills = ['Low', 'High']

    fig, ax = plt.subplots(figsize=(9, 5))

    for u in use_cases:
        for k in prompt_skills:
            test_rows = [
                profile | {H: float(h), 'Primary_Use_Case': u, 'Prompt_Engineering_Skill': k}
                for h in ai_hours_grid
            ]
            preds = pipeline.predict(pd.DataFrame(test_rows))
            ax.plot(ai_hours_grid, preds, label=f'{u} | Skill: {k}')

    ax.set_title('Phân tích Kịch bản What-if: Đáp ứng GPA dự đoán theo Giờ AI (Profile cố định)', fontsize=12)
    ax.set_xlabel('Số giờ dùng AI mỗi tuần (Weekly_GenAI_Hours)', fontsize=10)
    ax.set_ylabel('Điểm GPA dự đoán (Predicted GPA)', fontsize=10)
    ax.legend(fontsize=8, loc='best')
    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig, ax
