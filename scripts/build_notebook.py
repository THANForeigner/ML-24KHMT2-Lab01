"""Build the self-contained notebook. Analysis code lives in notebook cells, not imports."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))


md(r"""
# AI Impact on Students — Exploratory Data Analysis

**Lab 01 · Introduction to Machine Learning · Linear Regression**

Notebook phân tích dữ liệu thực trong `archive/`, xuất artifacts có thể truy vết và chuẩn bị thiết kế hồi quy dự đoán `Post_Semester_GPA`.
Nhận xét định lượng được sinh từ kết quả chạy; không lấy thống kê của tác giả làm kết quả của nhóm.

**Data contract:** structural audit trên toàn file → deterministic cleaning → frozen 70/15/15 split → EDA trên **train** → so sánh thăm dò trên **validation**.
**Test chỉ được dùng cho counts và integrity**, không dùng target test, fit, diagnostics hay lựa chọn mô hình.

Chạy **Restart Kernel → Run All**. Thay đường dẫn và tùy chọn ở cell cấu hình bên dưới. CSV gốc không bị sửa.
""")
code(r"""
# Centralized configuration — edit this cell only for paths and analysis policy.
import os
from pathlib import Path

_candidates = [Path.cwd(), *Path.cwd().parents]
PROJECT_ROOT = next((p for p in _candidates if (p / 'brief.md').is_file()), Path.cwd())
DATA_PATH = os.environ.get('LAB_DATA_PATH') or None  # None: discover CSV in archive/, data/raw/, /kaggle/input.
OUTPUT_DIR = Path(os.environ.get('LAB_OUTPUT_DIR', str(PROJECT_ROOT / 'reports')))
SPLIT_DIR = Path(os.environ.get('LAB_SPLIT_DIR', str(PROJECT_ROOT / 'data' / 'splits')))
SOURCE_METADATA_PATH = PROJECT_ROOT / 'data' / 'source_metadata.json'
TARGET_COLUMN = 'Post_Semester_GPA'
ID_COLUMN = 'Student_ID'
RANDOM_STATE = 42
TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
PLOT_SAMPLE_SIZE = 5000
FIGURE_DPI = 300
SAVE_FIGURES, SAVE_TABLES = True, True
RUN_OPTIONAL_TESTS = False
PLOT_VALIDATION_TARGET = False
MIN_GROUP_SIZE = 30
GPA_EQUAL_TOLERANCE = 1e-9  # floating-point equality only; measurement precision unverified.
HIGH_CORRELATION_THRESHOLD = 0.8
PREDICTION_CUTOFF = 'Before end-of-semester GPA is known'
SOURCE_URL = 'https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students'
# Only pre-semester GPA is explicitly located before the cutoff by the source.
VERIFIED_AVAILABLE = ['Pre_Semester_GPA']
# Source documents only an upper GPA endpoint; do not invent a floor.
GPA_UPPER_BOUNDARY = 4.0
GPA_LOWER_BOUNDARY = None
CONFIRMED_HOURS_ADDITIVE = False
assert all(0 < r < 1 for r in [TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO])
assert abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1) < 1e-10
assert PLOT_SAMPLE_SIZE > 0 and MIN_GROUP_SIZE > 0 and GPA_EQUAL_TOLERANCE >= 0
""")
md("""
## Execution helpers

Các hàm lưu output luôn đăng ký scope và trạng thái trong manifest. Rerun cùng dữ liệu/cấu hình được phép;
nếu fingerprint hoặc cấu hình thay đổi, cần chọn thư mục output/split mới để bảo vệ kết quả cũ.
Sampling chỉ áp dụng cho hình; bảng thống kê sử dụng toàn bộ train hợp lệ.
""")
code(r"""
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style='whitegrid', context='notebook', palette='colorblind')
plt.rcParams.update({'figure.dpi': 100, 'axes.titlesize': 10, 'axes.labelsize': 9})
pd.set_option('display.max_columns', 14)
pd.set_option('display.precision', 4)

TABLE_TITLES = [
 'Dataset Overview', 'Data Dictionary', 'Feature Type Summary', 'Unique Values Summary',
 'Missing Values Summary', 'Duplicate Records Summary', 'Invalid Values Summary', 'Outlier Summary',
 'Data Quality Report', 'Numerical Descriptive Statistics', 'Categorical Descriptive Statistics',
 'Category Frequency Distribution', 'Distribution Characteristics and Skewness', 'Target Statistics',
 'GPA Change Statistics', 'Improvement Summary', 'GPA Change by Initial GPA Group',
 'GPA Boundary Concentration', 'Pearson with Target', 'Spearman with Target',
 'Categorical vs Target Summary', 'Group Difference Summary', 'AI Usage Intensity Groups',
 'Academic AI Group Summary', 'Interaction Analysis Summary', 'Subgroup Summary',
 'Highly Correlated Predictor Pairs', 'VIF Analysis', 'Exploratory Regression Results',
 'Coefficients', 'Regression Assumption Diagnostics', 'Influential Observations Summary',
 'Derived Feature Definitions', 'Derived Feature Statistics', 'Original vs Transformed Comparison',
 'Encoding Recommendations', 'Scaling Recommendations', 'Feature Selection Recommendations',
 'Leakage Risk Assessment', 'Feature Availability Matrix', 'Split Integrity Report',
 'Dataset Bias and Limitations', 'Research Questions and Evidence', 'Key EDA Findings',
 'Feature Decision Matrix', 'Preprocessing Plan', 'Linear Regression Model Recommendations',
 'Dataset Limitations']
FIGURE_TITLES = [
 'Feature Types Distribution', 'Missing Percentage', 'Missing Pattern', 'Numerical Boxplots',
 'Outlier Percentage by Feature', 'Numerical Histograms', 'Continuous Density Plots',
 'Numerical Boxplots', 'Categorical Count Plots', 'Ordinal Distributions', 'Skewness Comparison',
 'Post GPA Distribution', 'Pre vs Post GPA Distribution', 'GPA Change Distribution',
 'Improved Unchanged Declined', 'Pre vs Post GPA with Identity Line', 'GPA Change by Initial GPA Group',
 'Numerical Predictors vs Post GPA', 'Numerical Predictors vs GPA Change', 'Pearson Heatmap',
 'Spearman Heatmap', 'Feature Correlation with Post GPA', 'Feature Correlation with GPA Change',
 'AI Hours vs GPA Linear and Binned Trend', 'Post GPA by Major', 'Post GPA by Year',
 'Post GPA by Primary Use Case', 'Post GPA by Prompt Skill', 'Post GPA by Paid Subscription',
 'Post GPA by Institutional Policy', 'GPA Change by Categorical Groups', 'GPA Change by AI Usage Hours',
 'AI Hours vs GPA by Prompt Skill', 'AI Hours vs GPA by Major', 'AI Hours and Study Heatmap',
 'Use Case and Prompt Skill Heatmap', 'Dependency and Anxiety Heatmap', 'AI Hours vs Change by Pre GPA',
 'AI Hours Dependency and Skill Retention', 'Predictor Correlation Heatmap', 'VIF Comparison',
 'Actual vs Predicted GPA', 'Residual vs Fitted', 'Residual Distribution', 'QQ Plot',
 'Scale Location', 'Cooks Distance', 'Residuals vs Key Predictors', 'Exploratory Model Comparison',
 'Original vs Transformed AI Hours', 'Derived Feature Distributions', 'Derived Feature Correlations',
 'Linear vs Polynomial Relationship', 'Correlation Before After Engineering', 'Sample Counts Across Splits',
 'Target Distribution Train vs Validation', 'Feature Availability and Leakage Risk Matrix']
T_SECTIONS = [2]*4 + [3]*5 + [4]*4 + [6]*5 + [7]*4 + [8]*4 + [9]*6 + [10]*6 + [11]*4 + [12]*6
F_SECTIONS = [2] + [3]*4 + [5]*6 + [6]*6 + [7]*14 + [8]*8 + [9]*10 + [10]*5 + [11]*3
assert len(TABLE_TITLES) == 48 and len(FIGURE_TITLES) == 57
assert len(T_SECTIONS) == 48 and len(F_SECTIONS) == 57
manifest = {}
tables = {}
warnings_log = []
for prefix, titles, sections, kind in [('T', TABLE_TITLES, T_SECTIONS, 'table'), ('F', FIGURE_TITLES, F_SECTIONS, 'figure')]:
    for i, (title, section) in enumerate(zip(titles, sections), 1):
        aid = f'{prefix}{i:02}'
        manifest[aid] = dict(artifact_id=aid, artifact_type=kind, title=title, relative_path='',
            section=section, data_scope='train', status='pending', related_ids='', reason='')

def note(text):
    display(Markdown(text))

def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')

def filename(aid, extension):
    slug = re.sub(r'[^a-z0-9]+', '_', manifest[aid]['title'].lower()).strip('_')
    return f'{aid}_{slug}.{extension}'

def skip(aid, reason, scope='train'):
    manifest[aid].update(status='skipped', reason=reason, data_scope=scope, relative_path='')

def save_table(aid, frame, scope='train', caption=None):
    frame = pd.DataFrame(frame)
    tables[aid] = frame.copy()
    if frame.shape[1] == 0:
        skip(aid, 'No applicable rows or columns for this optional analysis', scope)
        note(f'**{aid}** skipped: no applicable rows/columns.')
        return frame
    note(f"**{aid} · {manifest[aid]['title']}** — scope: `{scope}`; {len(frame):,} rows. " + (caption or ''))
    display(frame.head(12))
    if SAVE_TABLES:
        path = OUTPUT_DIR / 'tables' / filename(aid, 'csv')
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding='utf-8', float_format='%.17g')
        manifest[aid].update(status='generated', data_scope=scope, relative_path=path.relative_to(OUTPUT_DIR).as_posix())
    else:
        skip(aid, 'SAVE_TABLES=False; displayed inline only', scope)
    return frame

def save_figure(aid, fig, scope='train', caption=None):
    fig.suptitle(f"{aid} · {manifest[aid]['title']} [{scope}]", fontsize=12, y=1.02)
    fig.tight_layout()
    if SAVE_FIGURES:
        path = OUTPUT_DIR / 'figures' / filename(aid, 'png')
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
        manifest[aid].update(status='generated', data_scope=scope, relative_path=path.relative_to(OUTPUT_DIR).as_posix())
    else:
        skip(aid, 'SAVE_FIGURES=False; displayed inline only', scope)
    plt.show()
    plt.close(fig)
    if caption:
        note(caption)

def reuse(aid, source):
    record = manifest[source]
    if record['status'] in ['generated', 'reused']:
        manifest[aid].update(status='reused', data_scope=record['data_scope'], relative_path=record['relative_path'],
                            related_ids=source, reason=f'Equivalent panels shared with {source}')
    else:
        skip(aid, f'Alias {source}: {record["reason"]}', record['data_scope'])

def axes_grid(n, columns=3, height=3.1):
    columns = min(columns, max(1, n))
    fig, ax = plt.subplots(math.ceil(max(1,n)/columns), columns, figsize=(5*columns, height*math.ceil(max(1,n)/columns)), squeeze=False)
    for a in ax.flat[n:]:
        a.set_visible(False)
    return fig, list(ax.flat)

def plot_sample(frame):
    return frame.sample(min(len(frame), PLOT_SAMPLE_SIZE), random_state=RANDOM_STATE)

def safe_correlation(frame, x, y, method):
    pair = frame[[x,y]].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan).dropna()
    n = len(pair)
    reason = 'insufficient pairs' if n < 3 else ('constant feature or target' if pair.nunique().min() < 2 else '')
    value = np.nan if reason else pair[x].corr(pair[y], method=method)
    return {'feature':x, 'outcome':y, 'method':method, 'valid_n':n, 'coefficient':value, 'reason':reason}

def ordered_levels(series, feature=None):
    if isinstance(series.dtype, pd.CategoricalDtype):
        levels=[str(v) for v in series.cat.categories]
    elif feature in globals().get('orders',{}):
        documented=orders[feature]
        seen=series.dropna().astype(str).unique().tolist()
        levels=[v for v in documented if v in seen]+sorted(set(seen)-set(documented))
    elif pd.api.types.is_numeric_dtype(series):
        levels=[str(v) for v in sorted(series.dropna().unique())]
    else:
        levels=sorted(series.dropna().astype(str).unique())
    return levels+(['<missing>'] if series.isna().any() else [])

def summary_groups(frame, group, outcome=TARGET_COLUMN):
    labels=frame[group].astype('string').fillna('<missing>')
    part = frame.assign(_group=pd.Categorical(labels,categories=ordered_levels(frame[group],group),ordered=True))
    out = part.groupby('_group', observed=True, sort=True)[outcome].agg(['size','count','mean','median','std']).reset_index()
    out = out.rename(columns={'_group':'group','count':'valid_n','size':'group_n'})
    out.insert(0, 'feature', group)
    out['outcome'] = outcome
    out['se'] = out['std'] / np.sqrt(out['valid_n'])
    critical = stats.t.ppf(.975, (out['valid_n']-1).clip(lower=1))
    out['ci95_low'] = out['mean'] - critical*out['se']
    out['ci95_high'] = out['mean'] + critical*out['se']
    out['small_group'] = out['valid_n'] < MIN_GROUP_SIZE
    out['ci_method'] = 't-based mean CI; independence not verified'
    return out

def heatmap(aid, matrix, caption=None):
    if matrix.empty:
        skip(aid, 'No applicable numeric columns')
        return
    fig, ax = plt.subplots(figsize=(max(7, .65*len(matrix.columns)), max(5,.5*len(matrix))))
    sns.heatmap(matrix, vmin=-1, vmax=1, cmap='vlag', center=0, annot=len(matrix)<12, fmt='.2f', ax=ax)
    save_figure(aid, fig, caption=caption)
""")
md(r"""
## Section 1 — Introduction

**Objective.** Dự đoán GPA cuối học kỳ trước khi biết kết quả; xem nhóm thuộc tính AI có bổ sung khả năng dự đoán so với Pre GPA và học tập truyền thống hay không.
Hồi quy phù hợp vì target liên tục; sai số MAE/RMSE và R² cần được xem cùng giới hạn miền GPA.

**Nguồn:** [Kaggle · AI Impact on Students](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students),
metadata chính thức đã kiểm tra ngày 2026-11-17, version 1, **synthetic**, CC0-1.0.
Snapshot tại `data/source_metadata.json`; phiên bản byte của CSV cục bộ chưa được publisher checksum xác thực.
Không có hai PDF tham chiếu trong workspace; notebook thực hiện theo `brief.md`, chưa xác minh nguyên văn quy định giảng viên.

| RQ | Research question |
|---|---|
| RQ01 | Schema, completeness và validity có phù hợp regression? |
| RQ02 | Pre GPA dự đoán Post GPA tốt đến đâu? |
| RQ03 | AI Hours liên hệ thế nào với Post GPA và GPA Change? |
| RQ04 | Traditional Study Hours liên hệ thế nào với kết quả? |
| RQ05 | Kết quả có khác theo Prompt Skill, Use Case và nhóm sinh viên? |
| RQ06 | Dependency liên hệ với GPA, Change và Retention? |
| RQ07 | Có bằng chứng phi tuyến, bão hòa hoặc interaction? |
| RQ08 | AI features bổ sung khả năng dự đoán trên validation? |
| RQ09 | Có đa cộng tuyến hoặc phụ thuộc tuyến tính hoàn hảo? |
| RQ10 | Nên impute, encode, scale và engineer như thế nào? |
| RQ11 | OLS có giới hạn giả định, clipping và sai lệch nào? |
| RQ12 | Provenance và representativeness giới hạn khái quát ra sao? |

`GPA_Change = Post − Pre` chỉ là analytical outcome, **không được làm predictor**.
Association không chứng minh causation. E1 dùng Pre GPA là baseline trước kỳ;
E2/E3 dùng biến chưa xác minh thời điểm là **conditional / retrospective**, chưa đủ bằng chứng triển khai forecasting.
""")
md(r"""
## Section 2 — Dataset Overview

**Method.** Tìm CSV trong `archive/` trước; không giải nén lại và không tải mạng khi chạy.
Đọc UTF-8/BOM, giữ chuỗi gốc để kiểm toán coercion. Full-file audit chỉ dùng schema, missing, uniqueness, validity và thông tin chia tập;
min/max/categories phân phối trong Data Dictionary được tính từ **train** sau khi split.
""")
code(r"""
def discover_csv():
    if DATA_PATH:
        p = Path(DATA_PATH).expanduser()
        p = p if p.is_absolute() else PROJECT_ROOT / p
        if not p.is_file():
            raise FileNotFoundError(f'DATA_PATH does not exist: {p}')
        return p
    candidates = []
    for folder in [PROJECT_ROOT/'archive', PROJECT_ROOT/'data'/'raw', Path('/kaggle/input'), Path('/content')]:
        if folder.is_dir():
            candidates.extend(sorted(folder.rglob('*.csv')))
    valid = []
    for p in candidates:
        with p.open(encoding='utf-8-sig', newline='') as f:
            header = next(csv.reader(f), [])
        if TARGET_COLUMN in header:
            valid.append(p)
    if len(valid) != 1:
        raise ValueError(f'Set DATA_PATH explicitly; found {len(valid)} candidate CSVs: {[str(p) for p in valid]}. No synthetic fallback.')
    return valid[0]

csv_path = discover_csv()
raw_fingerprint = digest_file(csv_path)
raw = pd.read_csv(csv_path, encoding='utf-8-sig', dtype='string', keep_default_na=False)
if raw.empty:
    raise ValueError('CSV has no rows. Attach the actual dataset.')
if not {TARGET_COLUMN, 'Pre_Semester_GPA'}.issubset(raw.columns):
    raise ValueError('Core schema requires Post_Semester_GPA and Pre_Semester_GPA; configure the dataset explicitly.')
with csv_path.open(encoding='utf-8-sig', newline='') as f:
    original_header = next(csv.reader(f))
if len(original_header) != len(set(original_header)):
    raise ValueError('Duplicate CSV headers: resolve source schema explicitly before analysis.')

semantic = {
 'Student_ID':'identifier', 'Major_Category':'nominal', 'Year_of_Study':'ordinal',
 'Pre_Semester_GPA':'continuous', 'Weekly_GenAI_Hours':'continuous', 'Primary_Use_Case':'nominal',
 'Prompt_Engineering_Skill':'ordinal', 'Tool_Diversity':'discrete', 'Paid_Subscription':'binary',
 'Traditional_Study_Hours':'continuous', 'Perceived_AI_Dependency':'ordinal',
 'Institutional_Policy':'nominal', 'Anxiety_Level_During_Exams':'ordinal',
 'Post_Semester_GPA':'continuous', 'Skill_Retention_Score':'continuous', 'Burnout_Risk_Level':'ordinal'}
descriptions = dict(zip(semantic, [
 'Student identifier', 'Field of study', 'Academic standing', 'GPA at start of semester',
 'Average weekly generative AI usage', 'Main AI use purpose', 'Self-rated prompt proficiency',
 'Distinct AI tools used', 'Paid AI subscription', 'Weekly non-AI study time',
 'Self-rated AI dependency', 'Institutional AI policy', 'Exam anxiety self-report',
 'GPA at semester end', 'Post-semester skill retention', 'Assessed burnout risk']))
numeric_expected = ['Pre_Semester_GPA','Weekly_GenAI_Hours','Tool_Diversity','Traditional_Study_Hours',
                    'Perceived_AI_Dependency','Anxiety_Level_During_Exams',TARGET_COLUMN,'Skill_Retention_Score']
orders = {'Year_of_Study':['Freshman','Sophomore','Junior','Senior','Graduate'],
          'Prompt_Engineering_Skill':['Beginner','Intermediate','Advanced'],
          'Burnout_Risk_Level':['Low','Medium','High']}
auxiliary = ['Skill_Retention_Score','Burnout_Risk_Level']
units = {c:('hours/week' if 'Hours' in c else 'GPA points' if 'GPA' in c else
            'tools' if c=='Tool_Diversity' else 'points (0-100)' if c=='Skill_Retention_Score' else
            'ordinal score (1-10)' if c in ['Perceived_AI_Dependency','Anxiety_Level_During_Exams'] else 'category') for c in raw.columns}
source_meta = json.loads(SOURCE_METADATA_PATH.read_text(encoding='utf-8')) if SOURCE_METADATA_PATH.is_file() else {
    'provenance':'unverified', 'license':'unverified', 'dataset_version':'unverified'}
missing_optional = sorted(set(semantic)-set(raw.columns))
extra_columns = sorted(set(raw.columns)-set(semantic))
warnings_log.extend([f'Optional columns absent: {missing_optional}', f'Extra columns excluded from models: {extra_columns}'])

config_record = dict(target=TARGET_COLUMN, id=ID_COLUMN, seed=RANDOM_STATE,
    ratios=[TRAIN_RATIO,VALIDATION_RATIO,TEST_RATIO], plot_sample_size=PLOT_SAMPLE_SIZE, dpi=FIGURE_DPI,
    save_figures=SAVE_FIGURES, save_tables=SAVE_TABLES, optional_tests=RUN_OPTIONAL_TESTS,
    plot_validation_target=PLOT_VALIDATION_TARGET, min_group=MIN_GROUP_SIZE,
    gpa_tolerance=GPA_EQUAL_TOLERANCE, cutoff=PREDICTION_CUTOFF, verified_available=VERIFIED_AVAILABLE,
    high_correlation_threshold=HIGH_CORRELATION_THRESHOLD, gpa_upper=GPA_UPPER_BOUNDARY,
    gpa_lower=GPA_LOWER_BOUNDARY, hours_additive=CONFIRMED_HOURS_ADDITIVE,
    implementation_version='1.0', source_metadata_sha256=digest_file(SOURCE_METADATA_PATH) if SOURCE_METADATA_PATH.is_file() else None)
config_hash = hashlib.sha256(json.dumps(config_record, sort_keys=True).encode()).hexdigest()
run_signature = raw_fingerprint + ':' + config_hash
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
previous_metadata = OUTPUT_DIR/'run_metadata.json'
if previous_metadata.exists():
    prior = json.loads(previous_metadata.read_text(encoding='utf-8'))
    if prior.get('run_signature') != run_signature:
        raise ValueError('Output contains a different data/config run. Choose a NEW OUTPUT_DIR and SPLIT_DIR; prior run preserved.')
elif any(OUTPUT_DIR.iterdir()):
    raise ValueError('OUTPUT_DIR is nonempty without owned run_metadata.json; choose an empty directory.')
run_metadata = dict(run_signature=run_signature, status='running', started_at=datetime.now(timezone.utc).isoformat(),
    dataset_path=str(csv_path.resolve()), dataset_sha256=raw_fingerprint, config_sha256=config_hash,
    config=config_record, source_metadata=source_meta, row_count_raw=len(raw), columns_raw=len(raw.columns),
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','matplotlib','seaborn','scipy','scikit-learn','statsmodels','nbformat','nbclient']},
    python=platform.python_version(), warnings=warnings_log)
json_write(previous_metadata, run_metadata)

observed_dtype = pd.read_csv(csv_path, encoding='utf-8-sig', nrows=1000).dtypes.astype(str)
save_table('T01', pd.DataFrame([dict(rows=len(raw), columns=len(raw.columns), file_bytes=csv_path.stat().st_size,
    memory_bytes=int(raw.memory_usage(deep=True).sum()), sha256=raw_fingerprint,
    missing_optional='; '.join(missing_optional), extra_columns='; '.join(extra_columns),
    encoding='utf-8-sig', header_mapping='none', scope='full_structural_audit')]), 'full_structural_audit')
type_counts = pd.Series([semantic.get(c,'unknown') for c in raw.columns]).value_counts().rename_axis('semantic_type').reset_index(name='features')
save_table('T03', type_counts, 'full_structural_audit')
save_table('T04', pd.DataFrame({'feature':raw.columns, 'unique_including_blank':[raw[c].nunique(dropna=False) for c in raw],
    'loaded_dtype':[str(raw[c].dtype) for c in raw], 'inferred_dtype_first_1000':[observed_dtype.get(c,'unknown') for c in raw]}), 'full_structural_audit')
fig, ax = plt.subplots(figsize=(8,4))
ax.barh(type_counts.semantic_type, type_counts.features)
ax.set(xlabel='Number of features', ylabel='Semantic type')
save_figure('F01', fig, 'full_structural_audit')
info_buffer = io.StringIO()
raw.info(buf=info_buffer, memory_usage='deep')
print(info_buffer.getvalue())
print('CSV SHA-256:', raw_fingerprint)
note('**Interpretation.** T01/T03/T04 mô tả cấu trúc thực tế. Extra columns chỉ audit; thiếu optional sẽ skip phân tích phụ thuộc. '
     'Min/max và sample nội dung được giữ đến sau split. ID không phải predictor dù có dạng số.')
""")
md(r"""
## Section 3 — Data Quality Assessment and Frozen Splits

**Method.** Blank/whitespace → missing; không xem `0` là missing. Không tự nhận `NA`, `-1` là sentinel khi chưa có evidence.
Log trim/case normalization/coercion; invalid predictor → missing (impute sau split), invalid target → exclude.
GPA chỉ kiểm tra finite và upper boundary 4.0 được mô tả trong nguồn, chưa có floor đã xác minh.
Exact duplicates loại bản ghi lặp; cùng Student ID hoặc cùng nội dung ngoài ID được **group** trước split, không xóa các sinh viên khác ID.
Mọi row vẫn có stable key từ SHA-256 và vị trí dòng gốc; assignment chứa cả lý do exclusion.
""")
code(r"""
clean = raw.copy()
quality_log, invalid_log = [], []
missing_rows = []
for c in raw:
    blank = raw[c].str.strip().eq('')
    missing_rows.append(dict(feature=c, missing_count=int(blank.sum()), missing_pct=100*blank.mean(),
                             denominator=len(raw), sentinel_policy='none verified; zero retained'))
    trimmed = raw[c].str.strip()
    affected = int((trimmed != raw[c]).sum())
    quality_log.append(dict(feature=c, issue='surrounding whitespace', affected_rows=affected, severity='low',
        rule='strip surrounding whitespace only', action='correct' if affected else 'retain',
        reason='Semantics preserved; raw file unchanged', data_scope='full_structural_audit'))
    clean[c] = trimmed.mask(blank, pd.NA)
    if c in numeric_expected:
        parsed = pd.to_numeric(clean[c], errors='coerce').astype(float)
        invalid = clean[c].notna() & (parsed.isna() | ~np.isfinite(parsed))
        invalid_log.append(dict(feature=c, issue='numeric parse or nonfinite', affected_rows=int(invalid.sum()),
            rule='numeric and finite', action='exclude target / set predictor missing'))
        clean[c] = parsed.mask(invalid)
    elif c in semantic and semantic[c] in ['nominal','ordinal','binary']:
        # Resolve only case variants of documented/observed categories; preserve other labels.
        if c in orders:
            lookup = {s.casefold():s for s in orders[c]}
        elif c == 'Paid_Subscription':
            lookup = {'true':'True','false':'False'}
        else:
            values = clean[c].dropna().unique().tolist()
            lookup = {s.casefold():s for s in sorted(values, reverse=True)}
        mapped = clean[c].map(lambda v:lookup.get(v.casefold(),v) if pd.notna(v) else pd.NA)
        changed = (clean[c].fillna('<missing>') != mapped.fillna('<missing>'))
        quality_log.append(dict(feature=c, issue='case normalization', affected_rows=int(changed.sum()), severity='low',
            rule=json.dumps(lookup, ensure_ascii=False), action='correct' if changed.any() else 'retain',
            reason='Category mapping is logged explicitly', data_scope='full_structural_audit'))
        clean[c] = mapped.astype('string')

domains = {'Weekly_GenAI_Hours':(0,None), 'Traditional_Study_Hours':(0,None),
           'Tool_Diversity':(1,5), 'Perceived_AI_Dependency':(1,10),
           'Anxiety_Level_During_Exams':(1,10), 'Skill_Retention_Score':(0,100),
           'Pre_Semester_GPA':(GPA_LOWER_BOUNDARY,GPA_UPPER_BOUNDARY),
           TARGET_COLUMN:(GPA_LOWER_BOUNDARY,GPA_UPPER_BOUNDARY)}
integer_domains = ['Tool_Diversity','Perceived_AI_Dependency','Anxiety_Level_During_Exams']
for c, (lo,hi) in domains.items():
    if c not in clean:
        continue
    bad = pd.Series(False, index=clean.index)
    if lo is not None:
        bad |= clean[c] < lo
    if hi is not None:
        bad |= clean[c] > hi
    if c in integer_domains:
        bad |= clean[c].notna() & ~np.isclose(clean[c].fillna(0), clean[c].fillna(0).round())
    invalid_log.append(dict(feature=c, issue='domain violation', affected_rows=int(bad.sum()),
        rule=f'lower={lo}; upper={hi}; integer={c in integer_domains}; source=Kaggle metadata / nonnegative elapsed time',
        action='exclude target' if c==TARGET_COLUMN else 'set predictor missing; retain row'))
    clean.loc[bad,c] = np.nan
for c, allowed in {**orders,'Paid_Subscription':['True','False']}.items():
    if c in clean:
        bad = clean[c].notna() & ~clean[c].isin(allowed)
        invalid_log.append(dict(feature=c, issue='unknown documented category', affected_rows=int(bad.sum()),
            rule='documented categories: '+str(allowed), action='retain unknown category; one-hot; investigate'))

exact_mask = raw.duplicated(keep='first')
content_cols = [c for c in raw if c != ID_COLUMN]
duplicate_metrics = [
 dict(issue='exact duplicate', groups=int(raw[raw.duplicated(keep=False)].drop_duplicates().shape[0]),
      affected_rows=int(raw.duplicated(keep=False).sum()), removed_rows=int(exact_mask.sum()), action='remove subsequent exact records'),
 dict(issue='duplicate content excluding ID', groups=int(raw.loc[raw.duplicated(content_cols,keep=False),content_cols].drop_duplicates().shape[0]),
      affected_rows=int(raw.duplicated(content_cols,keep=False).sum()), removed_rows=0, action='retain different IDs; group connected content')]
if ID_COLUMN in clean:
    ids = clean[ID_COLUMN]
    repeated = ids.notna() & ids.duplicated(keep=False)
    conflicts = clean.loc[repeated].groupby(ID_COLUMN, dropna=False)[content_cols].nunique(dropna=False).gt(1).any(axis=1)
    duplicate_metrics.extend([
        dict(issue='repeated nonmissing Student ID',groups=int(ids[repeated].nunique()),affected_rows=int(repeated.sum()),removed_rows=0,action='group all records'),
        dict(issue='same ID conflicting content',groups=int(conflicts.sum()),affected_rows=int(ids.isin(conflicts[conflicts].index).sum()),removed_rows=0,action='retain and group; investigate')])
else:
    duplicate_metrics.append(dict(issue='Student ID absent',groups=0,affected_rows=0,removed_rows=0,action='content grouping only'))

row_keys = pd.Series([hashlib.sha256(f'{raw_fingerprint}:{i}'.encode()).hexdigest() for i in range(len(raw))], index=raw.index)
exclusion = pd.Series('', index=raw.index)
exclusion.loc[clean[TARGET_COLUMN].isna()] = 'missing_or_invalid_target'
exclusion.loc[exact_mask] = 'exact_duplicate'
eligible_idx = clean.index[exclusion.eq('')].to_numpy()
if len(eligible_idx) < 40:
    raise ValueError(f'Only {len(eligible_idx)} eligible rows; at least 40 required for split and exploratory regression.')

# Union-find joins shared IDs and shared content, including transitive connections; no N x N matrix.
parent = np.arange(len(raw))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return int(i)
def join_labels(indices, labels):
    first = {}
    for i, label in zip(indices, labels):
        if pd.isna(label):
            continue
        if label in first:
            a,b = find(int(i)),find(first[label])
            parent[max(a,b)] = min(a,b)
        else:
            first[label] = int(i)
if ID_COLUMN in clean:
    join_labels(eligible_idx, clean.loc[eligible_idx,ID_COLUMN])
content_hash = pd.util.hash_pandas_object(clean[content_cols], index=False).astype(str)
join_labels(eligible_idx, content_hash.loc[eligible_idx])
group_root = pd.Series({int(i):find(int(i)) for i in eligible_idx})
groups = group_root.to_numpy()
has_groups = len(set(groups)) < len(eligible_idx)
if has_groups:
    if len(set(groups)) < 7:
        raise ValueError('Too few independent groups to form usable train/validation/test. Revise split protocol explicitly.')
    splitter = GroupShuffleSplit(n_splits=1, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
    a,b = next(splitter.split(eligible_idx, groups=groups))
    train_idx, hold_idx = eligible_idx[a], eligible_idx[b]
    hold_groups = groups[b]
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=VALIDATION_RATIO/(VALIDATION_RATIO+TEST_RATIO), random_state=RANDOM_STATE)
    v,t = next(splitter2.split(hold_idx, groups=hold_groups))
    val_idx, test_idx = hold_idx[v],hold_idx[t]
else:
    train_idx, hold_idx = train_test_split(eligible_idx, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
    val_idx, test_idx = train_test_split(hold_idx, train_size=VALIDATION_RATIO/(VALIDATION_RATIO+TEST_RATIO), random_state=RANDOM_STATE)
if min(len(train_idx),len(val_idx),len(test_idx)) < 5:
    raise ValueError('A split has fewer than 5 rows; cannot run trustworthy exploratory comparisons.')
assignments = pd.DataFrame({'source_row_position':raw.index, 'row_key':row_keys, 'split':'excluded',
                            'exclusion_reason':exclusion, 'group_key':''})
for label, indices in [('train',train_idx),('validation',val_idx),('test',test_idx)]:
    assignments.loc[indices,'split'] = label
    assignments.loc[indices,'group_key'] = [row_keys.loc[group_root.loc[i]] for i in indices]
assignment_hash = hashlib.sha256(assignments.to_csv(index=False).encode()).hexdigest()
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
split_meta_path = SPLIT_DIR/'split_metadata.json'
if split_meta_path.exists():
    old = json.loads(split_meta_path.read_text(encoding='utf-8'))
    if old.get('dataset_sha256') != raw_fingerprint or old.get('split_config') != {
        'seed':RANDOM_STATE,'ratios':[TRAIN_RATIO,VALIDATION_RATIO,TEST_RATIO]} or old.get('assignment_sha256') != assignment_hash:
        raise ValueError('Existing split differs from data/config/algorithm. Choose NEW SPLIT_DIR; no silent overwrite.')
elif any(SPLIT_DIR.iterdir()):
    raise ValueError('Nonempty SPLIT_DIR without owned metadata; choose an empty split directory.')
assignments.to_csv(SPLIT_DIR/'split_assignment.csv', index=False, encoding='utf-8')
split_metadata = dict(dataset_sha256=raw_fingerprint, assignment_sha256=assignment_hash,
    split_config={'seed':RANDOM_STATE,'ratios':[TRAIN_RATIO,VALIDATION_RATIO,TEST_RATIO]},
    split_rule='group connected ID/content' if has_groups else 'random rows; no repeated groups',
    stable_key='sha256(dataset sha256 + colon + zero-based source row position)',
    content_group_method='pandas stable uint64 row content hash; independent of Python randomized hash',
    counts=assignments.split.value_counts().to_dict(), eligible_rows=len(eligible_idx), raw_rows=len(raw),
    actual_ratios={s:len(ix)/len(eligible_idx) for s,ix in [('train',train_idx),('validation',val_idx),('test',test_idx)]},
    exclusions=exclusion[exclusion.ne('')].value_counts().to_dict(), target_in_assignment=False)
json_write(split_meta_path, split_metadata)

# Materialize only train and validation data. Never create a test feature/target frame.
train = clean.loc[sorted(train_idx)].copy()
validation = clean.loc[sorted(val_idx)].copy()
if train[TARGET_COLUMN].nunique() < 2:
    raise ValueError('Training target is constant or missing: exploratory regression/R² is not meaningful. Check actual data and split.')
for part in [train, validation]:
    part['GPA_Change'] = part[TARGET_COLUMN] - part['Pre_Semester_GPA']
numeric_cols = [c for c in numeric_expected if c in train]
continuous_cols = [c for c in train if semantic.get(c)=='continuous']
categorical_cols = [c for c in train if semantic.get(c) in ['nominal','binary','ordinal'] and c not in numeric_cols]
ordinal_cols = [c for c in train if semantic.get(c)=='ordinal']
quantitative_predictors = [c for c in numeric_cols if c not in [TARGET_COLUMN,*auxiliary] and semantic.get(c)!='ordinal']
analysis_predictors = [c for c in numeric_cols if c not in [TARGET_COLUMN,*auxiliary]]
missing_table = save_table('T05', missing_rows, 'full_structural_audit')
save_table('T06', duplicate_metrics, 'full_structural_audit')
save_table('T07', invalid_log, 'full_structural_audit')
for r in invalid_log:
    quality_log.append(dict(feature=r['feature'],issue=r['issue'],affected_rows=r['affected_rows'],
        severity='high' if r['affected_rows'] else 'none', rule=r['rule'], action='investigate' if 'category' in r['issue'] else ('correct' if r['affected_rows'] else 'retain'),
        reason=r['action'],data_scope='full_structural_audit'))
quality_log.append(dict(feature='ALL',issue='regression sample exclusions',affected_rows=int(exclusion.ne('').sum()),severity='info',
    rule='no target imputation; exact duplicate removal only',action='remove',
    reason=f'before={len(raw)}; after={len(eligible_idx)}',data_scope='full_structural_audit'))
save_table('T09', quality_log, 'full_structural_audit')
dictionary = []
for c in raw:
    s = train[c]
    is_num = c in numeric_cols
    role = 'target' if c==TARGET_COLUMN else 'identifier' if c==ID_COLUMN else 'auxiliary outcome' if c in auxiliary else 'candidate predictor' if c in semantic else 'unknown; excluded'
    dictionary.append(dict(feature=c, description=descriptions.get(c,'unknown'), observed_dtype=str(s.dtype),
        loaded_dtype=str(raw[c].dtype), semantic_type=semantic.get(c,'unknown'), unit=units[c],
        verified_domain=str(domains.get(c,orders.get(c,'unverified'))),
        train_min=s.min() if is_num else np.nan, train_max=s.max() if is_num else np.nan,
        train_categories='<identifier values withheld>' if c==ID_COLUMN else '' if is_num else json.dumps(sorted(s.dropna().astype(str).unique()),ensure_ascii=False),
        structural_missing_count=int(missing_table.set_index('feature').loc[c,'missing_count']),
        train_missing_count=int(s.isna().sum()), train_unique=int(s.nunique()), role=role,
        availability='available' if c in VERIFIED_AVAILABLE else 'unavailable' if c in [ID_COLUMN,TARGET_COLUMN,*auxiliary] else 'unknown',
        verification_source=SOURCE_URL if c in semantic else 'unknown',
        data_scope='full_structural_audit (schema/missing); train (min/max/categories/unique)'))
save_table('T02', dictionary, 'train', 'Schema/missing is structural; observed ranges/categories are train only. ID sample values omitted.')
note(f'**Cleaning result.** Raw={len(raw):,}; eligible={len(eligible_idx):,}; excluded={exclusion.ne("").sum():,}. '
     f'Split: train={len(train_idx):,}, validation={len(val_idx):,}, test={len(test_idx):,}. '
     f'Grouped split={has_groups}. T06 phân biệt groups/affected rows. Các tỷ lệ thực tế nằm trong split_metadata.json.')
display(train.drop(columns=[ID_COLUMN],errors='ignore').head())
""")
code(r"""
fig, ax = plt.subplots(figsize=(9,5))
ax.barh(missing_table.feature, missing_table.missing_pct)
ax.set(xlabel='Missing / blank (%) of raw rows',ylabel='Feature')
save_figure('F02',fig,'full_structural_audit')
if train.isna().any().any():
    fig,ax=plt.subplots(figsize=(12,5))
    sns.heatmap(plot_sample(train.drop(columns=[ID_COLUMN],errors='ignore')).isna(),cbar=False,yticklabels=False,ax=ax)
    ax.set(xlabel='Feature',ylabel='Sampled training rows')
    save_figure('F03',fig,caption=f'Train-only sample, maximum n={PLOT_SAMPLE_SIZE}, seed={RANDOM_STATE}.')
else:
    skip('F03','No missing values in training data after documented cleaning')

outlier_rows=[]
for c in numeric_cols:
    s=train[c].dropna()
    q1,q3=s.quantile([.25,.75]) if len(s) else (np.nan,np.nan)
    iqr=q3-q1
    applicable=len(s)>3 and s.nunique()>2 and iqr>0 and semantic.get(c)=='continuous'
    lower,upper=q1-1.5*iqr,q3+1.5*iqr
    n_flag=int(((s<lower)|(s>upper)).sum()) if applicable else np.nan
    outlier_rows.append(dict(feature=c,valid_n=len(s),q1=q1,q3=q3,iqr=iqr,lower_fence=lower,upper_fence=upper,
        flagged_count=n_flag,flagged_pct=100*n_flag/len(s) if applicable else np.nan,
        p01=s.quantile(.01),p99=s.quantile(.99),
        reason='' if applicable else 'IQR not meaningful: discrete/ordinal, constant, missing, or IQR=0',action='retain; investigate; no automatic deletion'))
outlier_table=save_table('T08',outlier_rows)
fig,axes=axes_grid(len(numeric_cols))
for ax,c in zip(axes,numeric_cols):
    sns.boxplot(x=train[c],ax=ax,color='#56B4E9')
    ax.set(xlabel=f'{c} ({units[c]})',title=f'valid n={train[c].notna().sum():,}')
save_figure('F04',fig,caption='IQR fences are train-only heuristics. Valid extreme observations are retained.')
fig,ax=plt.subplots(figsize=(9,4))
valid_outliers=outlier_table.dropna(subset=['flagged_pct'])
ax.barh(valid_outliers.feature,valid_outliers.flagged_pct)
ax.set(xlabel='IQR flagged (%) of nonmissing training rows',ylabel='Continuous feature')
save_figure('F05',fig)
note(f'**Implication.** {int(outlier_table.flagged_count.fillna(0).sum()):,} feature-level IQR flags (không phải số sinh viên duy nhất). '
     'Giữ dữ liệu hợp lệ; imputer/encoder chỉ fit train. Không áp IQR máy móc cho ordinal/count hoặc biến có IQR=0.')
""")
md(r"""
## Section 4 — Descriptive Statistics

**Method.** Toàn bộ thống kê dưới đây chỉ dùng **train**. Numerical sample variance/std dùng `ddof=1`;
skewness là pandas adjusted Fisher–Pearson và kurtosis là **excess Fisher** (normal = 0).
Ordinal scales được trình bày thêm bằng tần suất; nominal không có mean.
Percentage của category dùng tổng số dòng train, bao gồm missing như một nhóm riêng.
""")
code(r"""
numeric_stats=[]
for c in numeric_cols:
    s=train[c].dropna()
    numeric_stats.append(dict(feature=c,semantic_type=semantic[c],valid_n=len(s),missing_pct=100*train[c].isna().mean(),
        mean=s.mean(),median=s.median(),mode=s.mode().iloc[0] if semantic[c]=='discrete' and len(s.mode()) else np.nan,
        std=s.std(ddof=1),variance=s.var(ddof=1),minimum=s.min(),maximum=s.max(),range=s.max()-s.min(),
        q1=s.quantile(.25),q3=s.quantile(.75),iqr=s.quantile(.75)-s.quantile(.25),
        p01=s.quantile(.01),p05=s.quantile(.05),p95=s.quantile(.95),p99=s.quantile(.99),
        skewness=s.skew(),excess_kurtosis=s.kurt(),
        limitation='ordinal numeric summaries assume coding distances; use frequencies for interpretation' if semantic[c]=='ordinal' else ''))
numerical_summary=save_table('T10',numeric_stats)
frequency_cols=list(dict.fromkeys(categorical_cols+ordinal_cols+[c for c in raw if c not in semantic]))
category_summary,frequency_rows=[],[]
for c in frequency_cols:
    counts=train[c].astype('string').fillna('<missing>').value_counts(dropna=False)
    category_summary.append(dict(feature=c,valid_n=int(train[c].notna().sum()),unique_nonmissing=int(train[c].nunique()),
        mode=str(counts.index[0]),mode_n=int(counts.iloc[0]),least_frequent=str(counts.index[-1]),least_n=int(counts.iloc[-1]),
        missing_pct=100*train[c].isna().mean(),rare_groups=int((counts<MIN_GROUP_SIZE).sum()),denominator=len(train)))
    frequency_rows.extend(dict(feature=c,category=str(level),count=int(n),percentage=100*n/len(train),
        denominator=len(train),small_group=bool(n<MIN_GROUP_SIZE)) for level,n in counts.items())
save_table('T11',category_summary)
frequencies=save_table('T12',frequency_rows,caption='Denominator = all training rows, including a missing group.')
save_table('T13',numerical_summary[['feature','valid_n','skewness','excess_kurtosis','std','iqr','p01','p99','limitation']])
note('**Interpretation.** T10–T13 là cơ sở cho missing handling và encoding. Mean của rating ordinal chỉ mô tả coding, '
     'không chứng minh khoảng cách tâm lý bằng nhau. Predictor không cần phân phối chuẩn để OLS hợp lệ.')
""")
md(r"""
## Section 5 — Univariate Analysis

**Method.** Continuous: histogram/KDE/boxplot. Discrete, ordinal và binary: frequency bar, theo thứ tự được mô tả trong nguồn khi có.
KDE chỉ chạy với biến liên tục có ít nhất ba giá trị phân biệt. Auxiliary outcomes được mô tả, không dùng làm predictors.
Không xóa biến chỉ vì skewness hay IQR flag; transformations là ứng viên cần kiểm chứng.
""")
code(r"""
fig,axes=axes_grid(len(continuous_cols))
for ax,c in zip(axes,continuous_cols):
    sns.histplot(train[c].dropna(),bins=40,ax=ax,color='#0072B2')
    ax.set(xlabel=f'{c} ({units[c]})',ylabel='Training rows',title=f'valid n={train[c].notna().sum():,}')
save_figure('F06',fig)
kde_cols=[c for c in continuous_cols if train[c].nunique()>=3 and train[c].notna().sum()>=10]
if kde_cols:
    fig,axes=axes_grid(len(kde_cols))
    for ax,c in zip(axes,kde_cols):
        sns.kdeplot(x=train[c].dropna(),ax=ax,cut=0)
        ax.set(xlabel=f'{c} ({units[c]})',ylabel='Estimated density')
    save_figure('F07',fig,caption='KDE is a descriptive smooth, not evidence of normality or a generating distribution.')
else:
    skip('F07','No continuous feature has sufficient distinct observations for KDE')
reuse('F08','F04')
nominal_plot_cols=[c for c in frequency_cols if c not in ordinal_cols]
if nominal_plot_cols:
    fig,axes=axes_grid(len(nominal_plot_cols),columns=2,height=3.5)
    for ax,c in zip(axes,nominal_plot_cols):
        counts=train[c].astype('string').fillna('<missing>').value_counts().head(20)
        ax.barh(counts.index.astype(str),counts.values)
        ax.set(xlabel='Training rows',ylabel=c,title=f'n={len(train):,}; top 20 if >20 levels')
    save_figure('F09',fig)
else:
    skip('F09','No nominal/binary/unknown feature')
ordered_plot_cols=list(dict.fromkeys(ordinal_cols+[c for c in numeric_cols if semantic[c]=='discrete']))
if ordered_plot_cols:
    fig,axes=axes_grid(len(ordered_plot_cols),columns=2,height=3.4)
    for ax,c in zip(axes,ordered_plot_cols):
        values=train[c].astype('string').fillna('<missing>')
        level_order=orders[c] if c in orders else sorted(values[values.ne('<missing>')].unique(), key=lambda v:float(v))
        level_order=list(level_order)+[v for v in values.unique() if v not in level_order]
        counts=values.value_counts().reindex(level_order,fill_value=0)
        ax.bar(counts.index.astype(str),counts.values)
        ax.tick_params(axis='x',rotation=25)
        ax.set(xlabel=c,ylabel='Training rows')
    save_figure('F10',fig)
else:
    skip('F10','No ordinal or discrete columns')
fig,ax=plt.subplots(figsize=(9,5))
ax.barh(numerical_summary.feature,numerical_summary.skewness)
ax.axvline(0,color='black',lw=.8)
ax.set(xlabel='Adjusted Fisher-Pearson sample skewness',ylabel='Feature')
save_figure('F11',fig)
for c in raw:
    if c==ID_COLUMN:
        continue
    missing_pct=100*train[c].isna().mean()
    if c in numeric_cols:
        s=train[c].dropna()
        role='auxiliary outcome, exclude from X' if c in auxiliary else 'primary target' if c==TARGET_COLUMN else 'candidate; confirm availability'
        note(f'**{c}:** valid n={len(s):,}, median={s.median():.4g}, IQR={s.quantile(.75)-s.quantile(.25):.4g}, '
             f'skew={s.skew():.3f}, missing={missing_pct:.3f}%. Implication: {role}; '
             + ('one-hot ordinal coding; no automatic log transform.' if semantic[c]=='ordinal' else 'retain valid extremes; median imputation fit on train if needed.'))
    else:
        counts=train[c].astype('string').fillna('<missing>').value_counts()
        note(f'**{c}:** {train[c].nunique()} nonmissing levels; largest group `{counts.index[0]}` '
             f'n={counts.iloc[0]:,} ({100*counts.iloc[0]/len(train):.2f}%), missing={missing_pct:.3f}%. '
             'Implication: one-hot for candidates; auxiliary/unknown columns excluded; flag rare groups. [T11/T12]')
""")
md(r"""
## Section 6 — Target Variable Analysis

**Method.** Post GPA và paired comparison chỉ trên train. `GPA_Change` dùng những dòng có cả hai GPA hợp lệ.
Improved/unchanged/declined dùng tolerance `1e-9` để chống sai số float, chưa phải ngưỡng thay đổi có ý nghĩa học thuật.
Nguồn ghi upper endpoint 4.0; không tự giả định floor. Pre GPA groups dùng quartiles train, ghi edges và bỏ duplicate edges.
GPA Change liên hệ cơ học với Pre GPA; cần cân nhắc mathematical coupling, regression to the mean và ceiling.
""")
code(r"""
target_stats=train[TARGET_COLUMN].describe().rename_axis('statistic').reset_index(name='value')
save_table('T14',target_stats)
paired=train[['Pre_Semester_GPA',TARGET_COLUMN,'GPA_Change']].dropna()
change=paired.GPA_Change
change_stats=change.describe().rename_axis('statistic').reset_index(name='value')
change_stats.loc[len(change_stats)]=['sample_std_ddof1',change.std(ddof=1)]
for method in ['pearson','spearman']:
    record=safe_correlation(paired,'Pre_Semester_GPA',TARGET_COLUMN,method)
    change_stats.loc[len(change_stats)]=[f'paired_{method}',record['coefficient']]
change_stats.loc[len(change_stats)]=['paired_mean_pre',paired.Pre_Semester_GPA.mean()]
change_stats.loc[len(change_stats)]=['paired_mean_post',paired[TARGET_COLUMN].mean()]
save_table('T15',change_stats)
improvement=pd.Series(np.select([change>GPA_EQUAL_TOLERANCE,change < -GPA_EQUAL_TOLERANCE],['Improved','Declined'],default='Unchanged'),index=change.index)
improvement_counts=improvement.value_counts().reindex(['Improved','Unchanged','Declined'],fill_value=0)
save_table('T16',pd.DataFrame({'status':improvement_counts.index,'count':improvement_counts.values,
    'percentage':100*improvement_counts.values/len(change) if len(change) else np.nan,
    'denominator':len(change),'tolerance':GPA_EQUAL_TOLERANCE}))

bin_log=[]
def quantile_bins(frame,column,label,q=4):
    if frame[column].dropna().nunique()<2:
        warnings_log.append(f'{label}: fewer than 2 distinct values; bins undefined')
        return pd.Series(pd.NA,index=frame.index,dtype='string')
    binned,edges=pd.qcut(frame[column],q=q,duplicates='drop',retbins=True)
    bin_log.append(dict(label=label,feature=column,edges=json.dumps(edges.tolist()),method='train qcut; duplicate edges dropped',valid_n=int(frame[column].notna().sum())))
    return binned
train['Pre_GPA_Group']=quantile_bins(train,'Pre_Semester_GPA','Pre GPA quartiles')
pre_groups=summary_groups(train,'Pre_GPA_Group','GPA_Change')
save_table('T17',pre_groups,caption='Quartiles fitted on train only; CI descriptive and assumes independence.')
boundary_rows=[]
for c in ['Pre_Semester_GPA',TARGET_COLUMN]:
    for label,boundary in [('upper',GPA_UPPER_BOUNDARY),('lower',GPA_LOWER_BOUNDARY)]:
        valid=train[c].dropna()
        n=int(np.isclose(valid,boundary,atol=GPA_EQUAL_TOLERANCE,rtol=0).sum()) if boundary is not None else np.nan
        boundary_rows.append(dict(feature=c,boundary_type=label,boundary=boundary,count=n,
            percentage=100*n/len(valid) if len(valid) and boundary is not None else np.nan,valid_n=len(valid),
            reason='source-documented upper endpoint' if boundary is not None else 'theoretical lower boundary unverified; not inferred from sample minimum'))
boundary_table=save_table('T18',boundary_rows)

fig,ax=plt.subplots(figsize=(8,4))
sns.histplot(train[TARGET_COLUMN],bins=40,ax=ax)
ax.set(xlabel='Post-semester GPA (points)',ylabel='Training rows')
save_figure('F12',fig)
fig,ax=plt.subplots(figsize=(8,4))
if len(paired):
    common_bins=np.histogram_bin_edges(paired[['Pre_Semester_GPA',TARGET_COLUMN]].to_numpy().ravel(),bins=40)
    ax.hist(paired.Pre_Semester_GPA,bins=common_bins,alpha=.5,label='Pre GPA')
    ax.hist(paired[TARGET_COLUMN],bins=common_bins,alpha=.5,label='Post GPA')
    ax.legend()
ax.set(xlabel='GPA (points), paired rows only',ylabel='Training rows')
save_figure('F13',fig)
fig,ax=plt.subplots(figsize=(8,4))
ax.hist(change,bins=40)
ax.axvline(0,color='black',ls='--')
ax.set(xlabel='Post - Pre GPA (points)',ylabel='Paired training rows')
save_figure('F14',fig)
fig,ax=plt.subplots(figsize=(7,4))
ax.bar(improvement_counts.index,improvement_counts.values)
ax.set(xlabel=f'Change class (tolerance {GPA_EQUAL_TOLERANCE:g})',ylabel='Paired training rows')
save_figure('F15',fig)
if len(paired):
    fig,ax=plt.subplots(figsize=(6,6))
    h=ax.hexbin(paired.Pre_Semester_GPA,paired[TARGET_COLUMN],gridsize=45,mincnt=1,cmap='viridis')
    limits=[paired[['Pre_Semester_GPA',TARGET_COLUMN]].min().min(),paired[['Pre_Semester_GPA',TARGET_COLUMN]].max().max()]
    ax.plot(limits,limits,'r--',label='Identity y=x')
    ax.set(xlim=limits,ylim=limits,xlabel='Pre GPA (points)',ylabel='Post GPA (points)',aspect='equal')
    fig.colorbar(h,ax=ax,label='Training rows per hexagon')
    ax.legend()
    save_figure('F16',fig)
else:
    skip('F16','No complete paired GPA records')
fig,ax=plt.subplots(figsize=(9,4))
ax.errorbar(pre_groups['group'],pre_groups['mean'],yerr=[pre_groups['mean']-pre_groups.ci95_low,pre_groups.ci95_high-pre_groups['mean']],fmt='o',capsize=4)
ax.axhline(0,color='black',ls='--')
ax.set(xlabel='Pre GPA training quartile (boundaries logged in T25)',ylabel='Mean GPA change (points)')
ax.tick_params(axis='x',rotation=20)
save_figure('F17',fig,caption='95% t-based CI; not a causal or independent effect of starting GPA.')
note(f'**Evidence.** Paired n={len(change):,}; mean change={change.mean():.4f}, median={change.median():.4f}, '
     f'sample std={change.std(ddof=1):.4f}. Improved={improvement_counts["Improved"]:,}, '
     f'unchanged={improvement_counts["Unchanged"]:,}, declined={improvement_counts["Declined"]:,}. '
     'Đây là thay đổi quan sát trong dữ liệu synthetic, chưa quy được cho AI. [T15/T16/T18]')
""")
md(r"""
## Section 7 — Bivariate Analysis

**Method.** Pearson cho quantitative; Spearman cho quantitative/ordinal có thứ tự đã được mô tả.
Nominal không được gán số để tính Pearson. Bảng ghi valid pair n và lý do N/A.
Predictor-only heatmaps loại ID, target và auxiliary outcomes. Auxiliary retention có bảng/hình riêng.
Group summaries báo n, mean, median, sample std và 95% t-based CI; CI có giả định independence chưa xác minh.
Không coi chênh lệch mean giữa nhóm là tác động AI; không screening chỉ bằng p-value.
""")
code(r"""
pearson_records=[]
spearman_records=[]
rank_data=train.copy()
rank_features=analysis_predictors.copy()
for c,levels in orders.items():
    if c in train and c not in auxiliary:
        rank_data[c]=train[c].map({v:i for i,v in enumerate(levels)}).astype(float)
        rank_features.append(c)
for outcome in [TARGET_COLUMN,'GPA_Change']:
    for c in quantitative_predictors:
        pearson_records.append(safe_correlation(train,c,outcome,'pearson'))
    for c in rank_features:
        spearman_records.append(safe_correlation(rank_data,c,outcome,'spearman'))
pearson_table=save_table('T19',pearson_records)
spearman_table=save_table('T20',spearman_records,caption='Documented ordinal order used for ranks only; no interval-distance assumption.')
group_cols=[c for c in dict.fromkeys(categorical_cols+ordinal_cols) if c not in auxiliary]
group_frames=[summary_groups(train,c,outcome) for c in group_cols for outcome in [TARGET_COLUMN,'GPA_Change']]
group_summary=pd.concat(group_frames,ignore_index=True) if group_frames else pd.DataFrame(columns=['feature','outcome','group','valid_n','mean','small_group'])
save_table('T21',group_summary)
group_differences=[]
for (feature,outcome),part in group_summary.groupby(['feature','outcome'],sort=False):
    valid=part.dropna(subset=['mean'])
    group_differences.append(dict(feature=feature,outcome=outcome,groups=len(valid),
        largest_mean_group=valid.loc[valid['mean'].idxmax(),'group'] if len(valid) else '',
        smallest_mean_group=valid.loc[valid['mean'].idxmin(),'group'] if len(valid) else '',
        mean_range=valid['mean'].max()-valid['mean'].min(),small_groups=int(valid.small_group.sum()),
        test_status='not run; optional tests disabled' if not RUN_OPTIONAL_TESTS else 'group tests deferred: no prespecified independent sampling design',
        limitation='unadjusted association; CIs descriptive, multiple comparisons not inferential'))
save_table('T22',group_differences)

def scatter_grid(aid,outcome,columns):
    if not columns:
        skip(aid,'No applicable numeric predictors')
        return
    fig,axes=axes_grid(len(columns))
    for ax,c in zip(axes,columns):
        pair=plot_sample(train[[c,outcome]].dropna())
        ax.scatter(pair[c],pair[outcome],s=5,alpha=.15,rasterized=True)
        ax.set(xlabel=f'{c} ({units.get(c,"rank")})',ylabel=outcome,title=f'displayed n={len(pair):,}')
    save_figure(aid,fig,caption=f'Plots sampled at most {PLOT_SAMPLE_SIZE:,}, seed {RANDOM_STATE}; correlations use all valid train pairs.')
scatter_grid('F18',TARGET_COLUMN,analysis_predictors)
scatter_grid('F19','GPA_Change',analysis_predictors)
predictor_corr=train[quantitative_predictors].corr(method='pearson',min_periods=3)
heatmap('F20',predictor_corr,'Predictors only; target and auxiliary outcomes excluded. Pair counts available from train missingness and T19 for target pairs.')
heatmap('F21',rank_data[rank_features].corr(method='spearman',min_periods=3))
for aid,outcome in [('F22',TARGET_COLUMN),('F23','GPA_Change')]:
    part=pearson_table[pearson_table.outcome.eq(outcome)].sort_values('coefficient')
    if len(part):
        fig,ax=plt.subplots(figsize=(8,4))
        ax.barh(part.feature,part.coefficient)
        ax.axvline(0,color='black',lw=.8)
        ax.set(xlabel=f'Pearson r with {outcome}',xlim=(-1,1),ylabel='Quantitative predictor')
        save_figure(aid,fig)
    else:
        skip(aid,'No applicable Pearson predictors')

H='Weekly_GenAI_Hours'
def binned_trend(frame,x,y,q=10,context=None):
    part=frame[[x,y]].dropna()
    if len(part)<3 or part[x].nunique()<2:
        return pd.DataFrame(columns=['x_mean','y_mean','n','bin_left','bin_right'])
    bins,edges=pd.qcut(part[x],q=q,duplicates='drop',retbins=True,labels=False)
    part=part.assign(_bin=bins)
    result=part.groupby('_bin',observed=True).agg(x_mean=(x,'mean'),y_mean=(y,'mean'),n=(y,'count')).reset_index()
    codes=result['_bin'].to_numpy(int)
    result['bin_left']=edges[codes]
    result['bin_right']=edges[codes+1]
    if context is not None:
        bin_log.append(dict(label=context,feature=x,edges=json.dumps(edges.tolist()),method=f'train qcut q={q}; duplicate edges dropped',valid_n=len(part)))
    return result.drop(columns='_bin')
if H in train and train[H].nunique()>1:
    fig,ax=plt.subplots(figsize=(8,5))
    pair=plot_sample(train[[H,TARGET_COLUMN]].dropna())
    ax.scatter(pair[H],pair[TARGET_COLUMN],s=6,alpha=.12,label=f'Sample n={len(pair):,}')
    all_pair=train[[H,TARGET_COLUMN]].dropna()
    slope,intercept=np.polyfit(all_pair[H],all_pair[TARGET_COLUMN],1)
    xline=np.linspace(all_pair[H].min(),all_pair[H].max(),100)
    ax.plot(xline,intercept+slope*xline,color='black',label='Linear fit (all valid train)')
    trend=binned_trend(train,H,TARGET_COLUMN,context='F24: marginal AI Hours vs Post GPA')
    ax.plot(trend.x_mean,trend.y_mean,'o-',color='#D55E00',label='Train decile binned mean')
    ax.set(xlabel='Weekly GenAI usage (hours/week)',ylabel='Post GPA (points)')
    ax.legend()
    save_figure('F24',fig,caption='Binned smoother avoids expensive full-data LOWESS. Marginal association, without academic adjustment.')
else:
    skip('F24','AI Hours absent or constant')
for aid,c in [('F25','Major_Category'),('F26','Year_of_Study'),('F27','Primary_Use_Case'),
              ('F28','Prompt_Engineering_Skill'),('F29','Paid_Subscription'),('F30','Institutional_Policy')]:
    if c not in train:
        skip(aid,f'Optional column absent: {c}')
        continue
    fig,ax=plt.subplots(figsize=(9,4))
    part=train[[c,TARGET_COLUMN]].dropna().copy()
    part[c]=part[c].astype(str)
    order=[v for v in orders.get(c,sorted(part[c].unique())) if v in set(part[c])]
    extras=sorted(set(part[c])-set(order))
    sns.boxplot(data=part,x=c,y=TARGET_COLUMN,order=order+extras,ax=ax,color='#56B4E9',fliersize=1)
    ax.tick_params(axis='x',rotation=20)
    ax.set(ylabel='Post GPA (points)',xlabel=c)
    save_figure(aid,fig,caption='Valid group n and small-group flags in T21; marginal comparison only.')
if group_cols:
    fig,axes=axes_grid(len(group_cols),columns=2,height=3.8)
    for ax,c in zip(axes,group_cols):
        part=group_summary[group_summary.feature.eq(c)&group_summary.outcome.eq('GPA_Change')]
        ax.errorbar(part['group'],part['mean'],yerr=[part['mean']-part.ci95_low,part.ci95_high-part['mean']],fmt='o',capsize=3)
        ax.set(xlabel=c,ylabel='Mean GPA change (95% t CI)')
        ax.tick_params(axis='x',rotation=25)
    save_figure('F31',fig)
else:
    skip('F31','No grouping predictors present')
for c in [H,'Traditional_Study_Hours','Perceived_AI_Dependency']:
    part=spearman_table[spearman_table.feature.eq(c)]
    if len(part):
        note('**'+c+'** — '+ '; '.join(f'{r.outcome}: Spearman={r.coefficient:.4f}, valid n={r.valid_n:,}' for r in part.itertuples())+
             '. [T20] Liên hệ marginal; chưa điều chỉnh Pre GPA và không chứng minh nhân quả.')
""")
md(r"""
## Section 8 — Multivariate Analysis

**Method.** Các quartiles/deciles được fit trên train, log edges và xử lý ties bằng `duplicates=drop`.
Grouped means/heatmaps báo count từng cell, mask nhóm dưới `MIN_GROUP_SIZE`; không tô 0 thay cell thiếu.
Xem AI × Prompt, Major, Study, Pre GPA, Dependency; Use Case × Prompt; Dependency × Anxiety và auxiliary Retention.
Phân tầng Pre GPA chỉ kiểm soát thô; conditional OLS ở Section 9 bổ sung kiểm soát tuyến tính, chưa phải causal moderation.
""")
code(r"""
multivariate_rows=[]
interaction_summaries=[]
if H in train:
    train['AI_Hours_Group']=quantile_bins(train,H,'AI Hours quartiles')
    ai_groups=summary_groups(train,'AI_Hours_Group','GPA_Change')
    save_table('T23',ai_groups)
else:
    ai_groups=pd.DataFrame()
    skip('T23','AI Hours absent')
if 'Traditional_Study_Hours' in train:
    train['Study_Hours_Group']=quantile_bins(train,'Traditional_Study_Hours','Traditional study quartiles')

def grouped_trend(aid,group,outcome):
    if H not in train or group not in train:
        skip(aid,f'Required columns unavailable: {H}, {group}')
        return
    fig,ax=plt.subplots(figsize=(9,5))
    for level,part in train.groupby(group,observed=True,dropna=False):
        trend=binned_trend(part,H,outcome,q=5,context=f'{aid}: {group}={level}; outcome={outcome}')
        multivariate_rows.extend(dict(analysis=f'{H} x {group}',group=str(level),outcome=outcome,
            hours_mean=r.x_mean,outcome_mean=r.y_mean,n=int(r.n),bin_left=r.bin_left,bin_right=r.bin_right,
            small_group=bool(r.n<MIN_GROUP_SIZE)) for r in trend.itertuples())
        eligible=trend.n.ge(MIN_GROUP_SIZE)
        ax.plot(trend.loc[eligible,'x_mean'],trend.loc[eligible,'y_mean'],'o-',label=f'{level} (n={len(part):,})')
    ax.set(xlabel='Weekly GenAI usage (hours/week)',ylabel=outcome)
    ax.legend(fontsize=8,loc='best')
    save_figure(aid,fig,caption='Within-group quintile means; small cells hidden. Bins computed only within train.')

if len(ai_groups):
    fig,ax=plt.subplots(figsize=(9,4))
    ax.errorbar(ai_groups['group'],ai_groups['mean'],yerr=[ai_groups['mean']-ai_groups.ci95_low,ai_groups.ci95_high-ai_groups['mean']],fmt='o',capsize=3)
    ax.set(xlabel='AI Hours train quartile',ylabel='Mean GPA change (95% t CI)')
    ax.tick_params(axis='x',rotation=20)
    save_figure('F32',fig)
else:
    skip('F32','AI Hours groups unavailable')
grouped_trend('F33','Prompt_Engineering_Skill',TARGET_COLUMN)
grouped_trend('F34','Major_Category',TARGET_COLUMN)
grouped_trend('F38','Pre_GPA_Group','GPA_Change')

def cross_heatmap(aid,left,right,outcome=TARGET_COLUMN):
    if left not in train or right not in train:
        skip(aid,f'Optional columns unavailable: {left}, {right}')
        return
    part=train[[left,right,outcome]].dropna().copy()
    if part.empty:
        skip(aid,'No complete training rows for cross-tabulation')
        return
    left_order=ordered_levels(part[left],left)
    right_order=ordered_levels(part[right],right)
    part[left],part[right]=part[left].astype(str),part[right].astype(str)
    grouped=part.groupby([left,right],observed=True)[outcome].agg(['count','mean','std']).reset_index()
    for r in grouped.to_dict('records'):
        interaction_summaries.append(dict(analysis=f'{left} x {right}',left_level=r[left],right_level=r[right],
            outcome=outcome,n=r['count'],mean=r['mean'],std=r['std'],small_group=r['count']<MIN_GROUP_SIZE))
    means=grouped.pivot(index=left,columns=right,values='mean').reindex(index=left_order,columns=right_order)
    counts=grouped.pivot(index=left,columns=right,values='count').reindex(index=left_order,columns=right_order)
    labels=means.map(lambda v:f'{v:.3f}' if pd.notna(v) else '')+'\nn='+counts.map(lambda v:str(int(v)) if pd.notna(v) else '')
    fig,ax=plt.subplots(figsize=(max(8,1.6*len(means.columns)),max(4,.7*len(means))))
    if counts.ge(MIN_GROUP_SIZE).any().any():
        sns.heatmap(means,mask=counts.lt(MIN_GROUP_SIZE)|means.isna(),annot=labels,fmt='',cmap='viridis',ax=ax,
                    cbar_kws={'label':f'Mean {outcome}'})
    else:
        ax.text(.5,.5,'All cells below minimum group size',ha='center',transform=ax.transAxes)
    ax.set(xlabel=right,ylabel=left)
    save_figure(aid,fig,caption=f'Cell annotations: mean and valid n. Cells n<{MIN_GROUP_SIZE} hidden; all counts retained in T24.')

cross_heatmap('F35','AI_Hours_Group','Study_Hours_Group')
cross_heatmap('F36','Primary_Use_Case','Prompt_Engineering_Skill')
cross_heatmap('F37','Perceived_AI_Dependency','Anxiety_Level_During_Exams')

# Auxiliary analysis is kept separate from predictor-only correlations and all X matrices.
retention='Skill_Retention_Score'
aux_records=[]
if retention in train:
    cols=[c for c in [H,'Perceived_AI_Dependency'] if c in train]
    if cols:
        fig,axes=axes_grid(len(cols),columns=2)
        for ax,c in zip(axes,cols):
            pair=plot_sample(train[[c,retention]].dropna())
            ax.scatter(pair[c],pair[retention],s=5,alpha=.15)
            ax.set(xlabel=f'{c} ({units[c]})',ylabel='Skill retention (0-100 points)')
            for method in ['pearson','spearman'] if c==H else ['spearman']:
                aux_records.append(safe_correlation(train,c,retention,method))
        save_figure('F39',fig,caption='Auxiliary outcome, never included in predictors. Statistics on all valid train rows, plots sampled.')
    else:
        skip('F39','AI Hours and Dependency both unavailable')
else:
    skip('F39','Auxiliary retention outcome absent')
if H in train and 'Perceived_AI_Dependency' in train:
    for level,part in train.groupby('Perceived_AI_Dependency',observed=True):
        for outcome in [TARGET_COLUMN,'GPA_Change']:
            rec=safe_correlation(part,H,outcome,'spearman')
            rec.update(analysis='AI Hours x Dependency',group=str(level))
            aux_records.append(rec)
if interaction_summaries:
    save_table('T24',interaction_summaries)
else:
    skip('T24','No applicable complete interaction cells')
save_table('T25',pd.DataFrame(bin_log),caption='Exact train quantile edges; ties use duplicate-edge removal. Trend-specific bin ranges recorded in T26.')
subgroup=pd.concat([pd.DataFrame(multivariate_rows),pd.DataFrame(aux_records)],ignore_index=True)
if len(subgroup):
    save_table('T26',subgroup,caption='Grouped trends plus separately labelled auxiliary/Dependency associations. n/coefficient missing where not applicable.')
else:
    skip('T26','No applicable grouped trends or auxiliary pairs')
note(f'**Evidence.** {len(interaction_summaries)} cross-tab cells và {len(multivariate_rows)} grouped-trend cells được xuất. '
     f'Ngưỡng diễn giải nhóm nhỏ n<{MIN_GROUP_SIZE}. Mô hình interaction chỉ là ứng viên; kiểm chứng cùng lower-order terms và trên validation/CV. [T24–T26]')
""")
md(r"""
## Section 9 — Regression-Oriented Analysis

### 9.1 Linearity and multicollinearity

**Method.** Preprocessing fit train: numeric median; categorical most-frequent + one-hot **drop first**.
Numeric ordinal ratings được one-hot mặc định, không ép equal spacing.
Loại all-missing hoặc encoded constants có log; reference categories và tên encoded term được giữ.
VIF tính từ correlation Gram matrix của design đã center, tương đương auxiliary regression có intercept;
constant/intercept không có VIF. Báo rank và condition number ở cả raw scale và standardized scale.
Ngưỡng VIF 5/10 và |r|≥0.8 chỉ là heuristic, không phải quy tắc xóa.

### 9.2 Exploratory comparison protocol

E0a train-mean; E0b persistence (missing Pre → train median); E1 Pre-only OLS;
E2 Academic OLS; E3 Academic + AI OLS. Tất cả dùng cùng train population và validation rows.
E1 được xem là baseline trước kỳ theo ý nghĩa Pre GPA. E2/E3 là **conditional / retrospective** do timing của các biến thêm chưa xác minh.
Delta E2/E3 đo giá trị bổ sung trong cùng retrospective scenario, chưa chứng minh forecast benefit.
Fit đúng một bộ cấu hình định trước; không tìm kiếm/tuning dựa trên validation.
""")
code(r"""
predictor_candidates=[c for c in raw if c in semantic and c not in [ID_COLUMN,TARGET_COLUMN,*auxiliary]]
assert not set(predictor_candidates) & {ID_COLUMN,TARGET_COLUMN,'GPA_Change',*auxiliary}
pearson_model_predictors=[c for c in predictor_candidates if c in quantitative_predictors]
model_corr=train[pearson_model_predictors].corr(min_periods=3)
high_pairs=[]
for i,left in enumerate(model_corr.columns):
    for right in model_corr.columns[i+1:]:
        value=model_corr.loc[left,right]
        if pd.notna(value) and abs(value)>=HIGH_CORRELATION_THRESHOLD:
            high_pairs.append(dict(left=left,right=right,pearson=value,valid_n=int(train[[left,right]].dropna().shape[0]),threshold=HIGH_CORRELATION_THRESHOLD))
save_table('T27',pd.DataFrame(high_pairs,columns=['left','right','pearson','valid_n','threshold']),
           caption='Empty table means no quantitative predictor pair passes the configured threshold; ordinal pair checks in T20/T26.')
reuse('F40','F20')

def input_frame(frame,features):
    part=frame[features].copy()
    for c in features:
        if c in numeric_expected and semantic.get(c) not in ['ordinal','binary']:
            part[c]=pd.to_numeric(part[c],errors='raise')
        else:
            part[c]=part[c].astype(object).where(part[c].notna(),np.nan)
    return part

models={}
preprocessing_log=[]
metric_records=[]
coefficient_records=[]
y_train=train[TARGET_COLUMN].to_numpy(float)
y_validation=validation[TARGET_COLUMN].to_numpy(float)

def record_metrics(model_id,pred,scenario,features,note_text=''):
    mse=mean_squared_error(y_validation,pred)
    metric_records.append(dict(model=model_id,scenario=scenario,features='; '.join(features),
        valid_n=len(pred),MAE=mean_absolute_error(y_validation,pred),MSE=mse,RMSE=np.sqrt(mse),
        R2=r2_score(y_validation,pred),fit_scope='train',evaluation_scope='validation',
        adjusted_R2_train=np.nan,note=note_text))

def fit_ols(model_id,features,scenario):
    features=[c for c in features if c in train]
    usable=[c for c in features if train[c].notna().any()]
    for c in set(features)-set(usable):
        preprocessing_log.append(dict(model=model_id,feature=c,action='exclude all-missing predictor; no train median/mode'))
    if not usable:
        warnings_log.append(f'{model_id}: no usable predictor; skipped')
        return None
    numerical=[c for c in usable if c in numeric_expected and semantic.get(c) not in ['ordinal','binary']]
    categorical=[c for c in usable if c not in numerical]
    transformers=[]
    if numerical:
        transformers.append(('num',Pipeline([('impute',SimpleImputer(strategy='median'))]),numerical))
    if categorical:
        transformers.append(('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
            ('encode',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False))]),categorical))
    transformer=ColumnTransformer(transformers,remainder='drop',verbose_feature_names_out=True)
    x=transformer.fit_transform(input_frame(train,usable)).astype(float)
    xv=transformer.transform(input_frame(validation,usable)).astype(float)
    names=transformer.get_feature_names_out().astype(str)
    keep=np.ptp(x,axis=0)>1e-12
    for name in names[~keep]:
        preprocessing_log.append(dict(model=model_id,feature=name,action='exclude encoded constant'))
    x,xv,names=x[:,keep],xv[:,keep],names[keep]
    if x.shape[1]==0 or len(x)<=x.shape[1]+5:
        raise ValueError(f'{model_id}: insufficient train rows / varying predictors for OLS: n={len(x)}, p={x.shape[1]+1}')
    design=np.column_stack([np.ones(len(x)),x])
    design_val=np.column_stack([np.ones(len(xv)),xv])
    rank=int(np.linalg.matrix_rank(design))
    fit=sm.OLS(y_train,design,hasconst=True).fit(cov_type='HC3' if RUN_OPTIONAL_TESTS else 'nonrobust')
    prediction=fit.predict(design_val)
    references={}
    if categorical:
        encoder=transformer.named_transformers_['cat'].named_steps['encode']
        for c,levels in zip(categorical,encoder.categories_):
            references[c]=str(levels[0])
            preprocessing_log.append(dict(model=model_id,feature=c,action=f'one-hot drop first; reference={levels[0]}; unknown->all-zero with warning'))
    for c in numerical:
        imputer=transformer.named_transformers_['num'].named_steps['impute']
        value=imputer.statistics_[numerical.index(c)]
        preprocessing_log.append(dict(model=model_id,feature=c,action=f'train median={value:.17g}; no scaling for raw-unit OLS'))
    record_metrics(model_id,prediction,scenario,usable)
    metric_records[-1]['adjusted_R2_train']=float(fit.rsquared_adj) if rank==design.shape[1] else np.nan
    ci=np.asarray(fit.conf_int())
    for j,term in enumerate(['intercept',*names]):
        origin=next((c for c in sorted(usable,key=len,reverse=True) if str(term).split('__')[-1]==c or str(term).split('__')[-1].startswith(c+'_')),'intercept')
        coefficient_records.append(dict(model=model_id,scenario=scenario,term=term,original_feature=origin,
            reference=references.get(origin,''),coefficient=fit.params[j],
            standard_error=fit.bse[j] if rank==design.shape[1] else np.nan,
            ci95_low=ci[j,0] if rank==design.shape[1] else np.nan,ci95_high=ci[j,1] if rank==design.shape[1] else np.nan,
            coefficient_unit='GPA points per '+units.get(origin,'indicator unit'),
            inference='classical OLS CI assumes independent homoscedastic errors' if not RUN_OPTIONAL_TESTS else 'HC3 robust CI; independence still required',
            identification='identified' if rank==design.shape[1] else 'rank deficient: pinv descriptive fit only; no coefficient inference'))
    result=dict(fit=fit,transformer=transformer,features=usable,names=names,x=x,design=design,design_val=design_val,
        val_prediction=prediction,rank=rank,scenario=scenario,references=references,keep=keep)
    models[model_id]=result
    return result

record_metrics('E0a',np.full(len(validation),y_train.mean()),'train-mean reference',[])
pre_median=train.Pre_Semester_GPA.median()
if pd.notna(pre_median):
    record_metrics('E0b',validation.Pre_Semester_GPA.fillna(pre_median).to_numpy(float),
        'pre-semester persistence',['Pre_Semester_GPA'],'Missing Pre imputed with train median, not validation statistics')
fit_ols('E1',['Pre_Semester_GPA'],'pre-semester baseline; deployment timing still requires institutional verification')
academic=[c for c in ['Pre_Semester_GPA','Major_Category','Year_of_Study','Traditional_Study_Hours'] if c in train]
fit_ols('E2',academic,'conditional / retrospective academic')
ai_additions=[c for c in ['Weekly_GenAI_Hours','Primary_Use_Case','Prompt_Engineering_Skill','Tool_Diversity',
    'Paid_Subscription','Perceived_AI_Dependency','Institutional_Policy','Anxiety_Level_During_Exams'] if c in train]
fit_ols('E3',list(dict.fromkeys(academic+ai_additions)),'conditional / retrospective academic + AI; exam anxiety high temporal risk')
results=pd.DataFrame(metric_records)
for mid,ref in [('E1','E0a'),('E2','E1'),('E3','E2')]:
    if mid in set(results.model) and ref in set(results.model):
        a=results.model.eq(mid)
        refrow=results.set_index('model').loc[ref]
        results.loc[a,'delta_reference']=ref
        for metric in ['MAE','RMSE','R2']:
            results.loc[a,'delta_'+metric]=results.loc[a,metric]-refrow[metric]
save_table('T29',results,'validation','Adjusted R² is TRAIN fitted-sample only; validation metrics are exploratory, never final test performance.')
coefficient_table=save_table('T30',coefficient_records)
note('**Modeling implication.** E2→E3 là so sánh incremental trong retrospective scenario trên cùng rows. '
     'E1→E2 thay đổi temporal assumptions nên delta không chứng minh cải thiện forecast. '
     'Raw coefficients khác đơn vị không phải feature importance. Reference groups, imputation và constant drops được lưu trong preprocessing_log.csv.')
""")
code(r"""
diagnostic_model=models.get('E3',models.get('E2',models.get('E1')))
if diagnostic_model is None:
    raise ValueError('No exploratory OLS model could be fitted; inspect target and Pre GPA.')
diagnostic_id=next(k for k,v in models.items() if v is diagnostic_model)
X=diagnostic_model['design']
x=diagnostic_model['x']
fit=diagnostic_model['fit']
n,p=X.shape
centered=x-x.mean(axis=0)
scaled=centered/centered.std(axis=0,ddof=1)
gram=scaled.T@scaled/(n-1)
condition_raw=float(np.linalg.cond(X))
condition_standardized=float(np.linalg.cond(np.column_stack([np.ones(n),scaled])))
vif_records=[]
for j,name in enumerate(diagnostic_model['names']):
    other=np.arange(x.shape[1])!=j
    if other.any():
        cross=gram[other,j]
        beta=np.linalg.lstsq(gram[np.ix_(other,other)],cross,rcond=None)[0]
        r_squared=float(cross@beta)
    else:
        r_squared=0.0
    vif=np.inf if 1-r_squared<=1e-10 else 1/(1-r_squared)
    origin=next((c for c in sorted(diagnostic_model['features'],key=len,reverse=True) if str(name).split('__')[-1]==c or str(name).split('__')[-1].startswith(c+'_')),'unknown')
    vif_records.append(dict(model=diagnostic_id,scenario=diagnostic_model['scenario'],encoded_term=name,original_feature=origin,VIF=vif,
        heuristic='high' if vif>10 else 'review' if vif>=5 else 'low',rank=diagnostic_model['rank'],columns_including_intercept=p,
        n=n,condition_number_raw=condition_raw,condition_number_standardized=condition_standardized,
        singular=diagnostic_model['rank']<p,method='correlation Gram auxiliary regression with intercept'))
vif_table=save_table('T28',vif_records)
fig,ax=plt.subplots(figsize=(10,max(5,.28*len(vif_table))))
finite=vif_table.VIF.replace(np.inf,np.nan)
cap=max(11,float(finite.max()) if finite.notna().any() else 11)
ax.barh(vif_table.encoded_term,vif_table.VIF.replace(np.inf,cap))
ax.axvline(5,color='orange',ls='--',label='5: review heuristic')
ax.axvline(10,color='red',ls='--',label='10: high heuristic')
ax.set(xlabel='VIF (infinite terms shown at chart cap)',ylabel='Encoded predictor')
ax.legend()
save_figure('F41',fig,caption=f'{diagnostic_id}; rank {diagnostic_model["rank"]}/{p}. VIF is term-specific; no invented feature importance score.')

fitted=np.asarray(fit.fittedvalues)
residual=y_train-fitted
leverage=np.einsum('ij,jk,ik->i',X,np.asarray(fit.normalized_cov_params),X)
mse_train=float(np.sum(residual**2)/(n-diagnostic_model['rank']))
denom=np.sqrt(mse_train*np.maximum(1-leverage,np.finfo(float).eps))
standardized=residual/denom
cooks=(residual**2/(diagnostic_model['rank']*mse_train))*leverage/np.maximum(1-leverage,np.finfo(float).eps)**2
diagnostic_records=[
 dict(check='mean specification',value=float(residual.mean()),method='train residual mean; inspect residual vs fitted and key predictors',limitation='zero mean with intercept does not prove linearity'),
 dict(check='residual variance',value=mse_train,method='SSE / residual df',limitation='inspect scale-location; heteroscedasticity possible'),
 dict(check='residual normality',value=float(pd.Series(residual).skew()),method='residual skewness + histogram + Q-Q',limitation='normality affects inference, not a predictor normality requirement'),
 dict(check='independence',value=np.nan,method='sampling/design audit; ID grouping before split',limitation='unknown sampling design and synthetic generator; residual plots cannot verify independence'),
 dict(check='rank',value=diagnostic_model['rank'],method=f'{p} design columns including intercept',limitation='pinv coefficients not identified if rank deficient'),
 dict(check='predictions above documented GPA endpoint',value=int((fitted>GPA_UPPER_BOUNDARY).sum()) if GPA_UPPER_BOUNDARY is not None else np.nan,
      method='unclipped train fitted predictions',limitation='no silent clipping; floor unknown'),
 dict(check='large internally standardized residual',value=int((np.abs(standardized)>3).sum()),method='abs(e / sqrt(MSE*(1-h))) > 3',limitation='heuristic; retain observations'),
 dict(check='high leverage',value=int((leverage>2*p/n).sum()),method=f'h > 2p/n; threshold={2*p/n:.8g}; n={n}; p={p}',limitation='heuristic, not an exclusion rule'),
 dict(check='large Cook distance',value=int((cooks>4/n).sum()),method=f'Cook D > 4/n; threshold={4/n:.8g}; rank used in Cook denominator',limitation='heuristic; inspect, do not delete automatically')]
if RUN_OPTIONAL_TESTS and diagnostic_model['rank']==p:
    lm,lmp,fvalue,fp=het_breuschpagan(residual,X)
    jb=stats.jarque_bera(residual)
    diagnostic_records.extend([
        dict(check='Breusch-Pagan',value=float(lmp),method=f'LM={lm:.8g}; p-value shown',limitation='sensitive at large n; no causal/sampling inference'),
        dict(check='Jarque-Bera',value=float(jb.pvalue),method=f'statistic={jb.statistic:.8g}; p-value shown',limitation='large-n sensitivity; inspect practical departure')])
else:
    diagnostic_records.append(dict(check='optional formal tests',value=np.nan,method='skipped: RUN_OPTIONAL_TESTS=False or singular design',limitation='core visual/effect-size diagnostics retained'))
diagnostics=save_table('T31',diagnostic_records)
influence=pd.DataFrame({'row_key':row_keys.loc[train.index].to_numpy(),'internally_standardized_residual':standardized,
    'leverage':leverage,'cooks_distance':cooks,'abs_residual_gt3':np.abs(standardized)>3,
    'leverage_gt_2p_n':leverage>2*p/n,'cooks_gt_4_n':cooks>4/n,'n':n,'p':p})
save_table('T32',influence.nlargest(min(30,n),'cooks_distance'),caption='Top 30 train rows by Cook D, identified only by stable keys. Flag totals in T31; no rows deleted.')
print(f'Diagnostic model: {diagnostic_id}; n={n:,}; p={p}; rank={diagnostic_model["rank"]}; no N x N hat matrix constructed.')
""")
md(r"""
### 9.3 Residual and influence diagnostics (train)

Residual = actual − fitted. Internally standardized residual dùng MSE và leverage; Cook's D tính bằng công thức vectorized,
không tạo hat matrix N×N. Các threshold chỉ để điều tra, không tự động loại observations.
Đồ thị residual theo predictor là kiểm tra conditional mean sau mô hình; không suy independence từ đồ thị.
Nếu heteroscedasticity/ceiling rõ, xem robust SE hoặc model specification khác trong training sau EDA.
""")
code(r"""
fig,ax=plt.subplots(figsize=(6,6))
h=ax.hexbin(fitted,y_train,gridsize=45,mincnt=1,cmap='viridis')
limits=[min(fitted.min(),y_train.min()),max(fitted.max(),y_train.max())]
ax.plot(limits,limits,'r--')
ax.set(xlabel=f'{diagnostic_id} fitted GPA',ylabel='Actual GPA',xlim=limits,ylim=limits,aspect='equal')
fig.colorbar(h,ax=ax,label='Training rows per hexagon')
save_figure('F42',fig)
diagnostic_frame=pd.DataFrame({'fitted':fitted,'residual':residual,'standardized':standardized,'leverage':leverage,'cooks':cooks},index=train.index)
shown=plot_sample(diagnostic_frame)
fig,ax=plt.subplots(figsize=(8,4))
ax.scatter(shown.fitted,shown.residual,s=5,alpha=.15)
ax.axhline(0,color='red',ls='--')
trend=binned_trend(diagnostic_frame,'fitted','residual')
ax.plot(trend.x_mean,trend.y_mean,'o-',color='black',label='All-train decile means')
ax.set(xlabel='Train fitted GPA',ylabel='Residual (actual - fitted)')
ax.legend()
save_figure('F43',fig)
fig,ax=plt.subplots(figsize=(8,4))
ax.hist(residual,bins=50)
ax.set(xlabel='Train residual (GPA points)',ylabel='Training rows')
save_figure('F44',fig)
fig,ax=plt.subplots(figsize=(6,5))
stats.probplot(standardized,dist='norm',plot=ax)
ax.set(title='Internally standardized residuals',xlabel='Theoretical normal quantiles',ylabel='Ordered residual quantiles')
save_figure('F45',fig)
fig,ax=plt.subplots(figsize=(8,4))
ax.scatter(shown.fitted,np.sqrt(np.abs(shown.standardized)),s=5,alpha=.15)
ax.set(xlabel='Train fitted GPA',ylabel='sqrt(abs(internally standardized residual))')
save_figure('F46',fig)
fig,ax=plt.subplots(figsize=(9,4))
ax.scatter(np.arange(n),cooks,s=3,alpha=.4)
ax.axhline(4/n,color='red',ls='--',label=f'4/n = {4/n:.3g}')
ax.set(xlabel='Training observation position (not Student ID)',ylabel="Cook's distance")
ax.legend()
save_figure('F47',fig)
residual_keys=[c for c in ['Pre_Semester_GPA',H,'Traditional_Study_Hours','Perceived_AI_Dependency'] if c in train]
fig,axes=axes_grid(len(residual_keys),columns=2)
for ax,c in zip(axes,residual_keys):
    part=pd.DataFrame({c:train[c],'residual':residual}).dropna()
    shown_part=plot_sample(part)
    ax.scatter(shown_part[c],shown_part.residual,s=5,alpha=.12)
    trend=binned_trend(part,c,'residual')
    ax.plot(trend.x_mean,trend.y_mean,'o-',color='black')
    ax.axhline(0,color='red',ls='--')
    ax.set(xlabel=c,ylabel='Train conditional-model residual')
save_figure('F48',fig)
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,metric in zip(axes,['MAE','RMSE']):
    ax.bar(results.model,results[metric])
    ax.set(xlabel='Prespecified exploratory model',ylabel=f'Validation {metric} (GPA points)')
save_figure('F49',fig,'validation',caption='Same validation rows; conditional models do not establish deployment forecast performance.')
note(f'**Diagnostics.** {diagnostic_id}: rank={diagnostic_model["rank"]}/{p}, '
     f'max finite VIF={vif_table.VIF.replace(np.inf,np.nan).max():.3f}, '
     f'|internally standardized residual|>3: {(np.abs(standardized)>3).sum():,}; '
     f'Cook D>4/n: {(cooks>4/n).sum():,}. [T28/T31/T32] '
     'Inspect F43/F46 cho mean specification/variance; các flag không chứng minh dữ liệu sai.')
""")
md(r"""
## Section 10 — Feature Engineering Exploration

**Method.** Transformations chỉ fit train; giữ hierarchy (H + H², main effects + interactions).
Centering dùng `mean_train(H)`; log1p cần H≥0; không ghép raw H và centered H cùng intercept.
Hours cùng đơn vị tuần nhưng chưa xác minh AI usage chỉ dành cho học và không overlap; **skip Total Study Hours/AI Study Ratio** mặc định.
Ordinal Dependency/Anxiety chưa được chứng minh equal spacing nên numeric-product interactions chỉ là proposal; prompt dùng reference-coded dummy interaction.
Một quadratic extension được định trước để so sánh thăm dò trên validation, không tìm kiếm degree/hyperparameters.
""")
code(r"""
derived=pd.DataFrame(index=train.index)
definitions=[
 dict(feature='GPA_Change',formula='Post_Semester_GPA - Pre_Semester_GPA',status='analytical outcome only',constraint='never predictor',fit_scope='no fitted parameters'),
 dict(feature='Total_Study_Hours',formula='H + Traditional_Study_Hours',status='skipped' if not CONFIRMED_HOURS_ADDITIVE else 'conditional',constraint='units + study-only coverage + no double count must be verified',fit_scope='train'),
 dict(feature='AI_Study_Ratio',formula='H / Total_Study_Hours',status='skipped' if not CONFIRMED_HOURS_ADDITIVE else 'conditional',constraint='valid total >0; zero denominator -> missing',fit_scope='train'),
 dict(feature='AI_Hours_Squared',formula='H**2',status='candidate',constraint='retain H; redundancy audit',fit_scope='train'),
 dict(feature='Log_AI_Hours',formula='log1p(H)',status='candidate',constraint='finite nonnegative H',fit_scope='train'),
 dict(feature='Centered_AI_Hours',formula='H - mean_train(H)',status='candidate',constraint='never use with raw H and intercept',fit_scope='train mean'),
 dict(feature='Centered_AI_Hours_Squared',formula='(H - mean_train(H))**2',status='candidate',constraint='retain H or centered H; no duplicate linear term',fit_scope='train mean'),
 dict(feature='AI_Dependency_Interaction',formula='H * Dependency',status='proposal only; skipped numeric product',constraint='ordinal interval-distance assumption and timing unverified',fit_scope='train'),
 dict(feature='AI_Prompt_Interaction',formula='centered H * reference-coded Prompt dummy',status='candidate',constraint='train reference; retain prompt dummies and H',fit_scope='train reference and mean'),
 dict(feature='Study_Anxiety_Interaction',formula='Study * Anxiety',status='proposal only; skipped numeric product',constraint='high temporal risk; interval-distance assumption unverified',fit_scope='train')]
if H in train:
    h=train[H]
    h_mean=h.mean()
    derived['AI_Hours_Squared']=h**2
    derived['Log_AI_Hours']=np.log1p(h.where(h>=0))
    derived['Centered_AI_Hours']=h-h_mean
    derived['Centered_AI_Hours_Squared']=(h-h_mean)**2
    if 'Prompt_Engineering_Skill' in train:
        levels=sorted(train.Prompt_Engineering_Skill.dropna().astype(str).unique())
        prompt_ref=levels[0] if levels else None
        for level in levels[1:]:
            derived['H_x_Prompt_'+level]=(h-h_mean)*(train.Prompt_Engineering_Skill.astype('string').eq(level).astype(float))
        preprocessing_log.append(dict(model='engineering',feature='AI_Prompt_Interaction',action=f'centered train H x prompt dummy; reference={prompt_ref}; retain main effects'))
    if CONFIRMED_HOURS_ADDITIVE and 'Traditional_Study_Hours' in train:
        derived['Total_Study_Hours']=h+train.Traditional_Study_Hours
        derived['AI_Study_Ratio']=h/derived.Total_Study_Hours.where(derived.Total_Study_Hours>0)
else:
    for row in definitions:
        if row['feature']!='GPA_Change':
            row['status']='skipped: AI Hours absent'
save_table('T33',definitions)
derived_stats=[]
for c in derived:
    s=derived[c].dropna()
    derived_stats.append(dict(feature=c,valid_n=len(s),mean=s.mean(),median=s.median(),std=s.std(ddof=1),
        minimum=s.min(),maximum=s.max(),missing_pct=100*derived[c].isna().mean(),skewness=s.skew(),
        pearson_post=derived[c].corr(train[TARGET_COLUMN]),spearman_post=derived[c].corr(train[TARGET_COLUMN],method='spearman'),
        pearson_change=derived[c].corr(train.GPA_Change),decision='candidate only; temporal assumptions and hierarchy required'))
if len(derived_stats):
    save_table('T34',derived_stats)
else:
    skip('T34','No valid engineered features; AI Hours absent')

transformation_comparison=[]
if H in train:
    for name,series in [('Original_H',train[H]),*[(c,derived[c]) for c in derived]]:
        valid=series.dropna()
        transformation_comparison.append(dict(candidate=name,valid_n=len(valid),skewness=valid.skew(),
            pearson_post=series.corr(train[TARGET_COLUMN]),spearman_post=series.corr(train[TARGET_COLUMN],method='spearman'),
            pair_vif_with_linear_H=np.nan,validation_MAE=np.nan,validation_RMSE=np.nan,validation_R2=np.nan,
            quadratic_coefficient=np.nan,reason='distribution/correlation only; not an automatic feature selection rule'))
    for c,linear in [('AI_Hours_Squared',train[H]),('Centered_AI_Hours_Squared',derived.Centered_AI_Hours)]:
        corr=derived[c].corr(linear)
        for row in transformation_comparison:
            if row['candidate']==c:
                row['pair_vif_with_linear_H']=np.inf if pd.notna(corr) and 1-corr**2<=1e-10 else 1/(1-corr**2) if pd.notna(corr) else np.nan

    # Fixed conditional quadratic extension of E3 (same train/validation population), no tuning.
    if 'E3' in models and train[H].notna().any():
        base=models['E3']
        train_h=train[H].fillna(train[H].median()).to_numpy(float)
        val_h=validation[H].fillna(train[H].median()).to_numpy(float)
        h_center=float(train_h.mean())
        xd=np.column_stack([base['design'],(train_h-h_center)**2])
        xdv=np.column_stack([base['design_val'],(val_h-h_center)**2])
        engineering_rank=int(np.linalg.matrix_rank(xd))
        if engineering_rank==xd.shape[1]:
            quadratic_fit=sm.OLS(y_train,xd).fit()
            quadratic_prediction=quadratic_fit.predict(xdv)
            base_result=results.set_index('model').loc['E3']
            for row in transformation_comparison:
                if row['candidate']=='Original_H':
                    row.update(validation_MAE=base_result.MAE,validation_RMSE=base_result.RMSE,validation_R2=base_result.R2,
                        reason='E3 conditional baseline, fixed feature set; validation exploratory')
                elif row['candidate']=='Centered_AI_Hours_Squared':
                    row.update(validation_MAE=mean_absolute_error(y_validation,quadratic_prediction),
                        validation_RMSE=np.sqrt(mean_squared_error(y_validation,quadratic_prediction)),
                        validation_R2=r2_score(y_validation,quadratic_prediction),quadratic_coefficient=float(quadratic_fit.params[-1]),
                        reason='Fixed E3 + centered H^2; retain H, no extra linear duplicate; conditional/retrospective; not tuned')
        else:
            warnings_log.append('Quadratic extension skipped: rank deficient; polynomial coefficient not identifiable')
if transformation_comparison:
    save_table('T35',transformation_comparison,'validation',
        'Skew/correlation/pair VIF are TRAIN; only explicitly populated performance fields use VALIDATION. No test data.')
else:
    skip('T35','No applicable AI Hours transformation')

encoding_rows=[]
scaling_rows=[]
selection_rows=[]
for c in raw:
    sem=semantic.get(c,'unknown')
    forbidden=c in [ID_COLUMN,TARGET_COLUMN,*auxiliary] or c in extra_columns
    encoding='exclude from X' if forbidden else 'median imputation' if sem in ['continuous','discrete'] else 'mode imputation + one-hot drop-first; unknown categories handled'
    encoding_rows.append(dict(feature=c,semantic_type=sem,encoding=encoding,fit_scope='train only',
        ordinal_policy='one-hot by default; ranks used only for Spearman' if sem=='ordinal' else '',
        reason='ID/target/auxiliary/unknown forbidden in predictors' if forbidden else 'preserve units; avoid invented ordinal distances'))
    scaling_rows.append(dict(feature=c,OLS='no scaling required; raw-unit coefficients' if not forbidden else 'not a predictor',
        polynomial_ridge='standardize encoded numeric matrix inside training pipeline' if not forbidden else 'not applicable',
        fit_scope='train only; never validation or test'))
    association=pearson_table[pearson_table.feature.eq(c)&pearson_table.outcome.eq(TARGET_COLUMN)]
    selection_rows.append(dict(feature=c,decision='Exclude' if forbidden else 'Keep' if c in VERIFIED_AVAILABLE else 'Conditional',
        rationale='role-based leakage guard' if forbidden else 'pre-semester availability' if c in VERIFIED_AVAILABLE else 'timing unverified; retrospective analysis only',
        pearson_post=association.coefficient.iloc[0] if len(association) else np.nan,
        evidence='T19; T20; T28; T33; T35; T40',screening_policy='no elimination solely by marginal correlation or skewness'))
save_table('T36',encoding_rows)
save_table('T37',scaling_rows)
save_table('T38',selection_rows)
if H in train and derived.shape[1]:
    fig,axes=axes_grid(3)
    for ax,c in zip(axes,[H,'Log_AI_Hours','Centered_AI_Hours_Squared']):
        s=train[c] if c in train else derived[c]
        ax.hist(s.dropna(),bins=40)
        ax.set(xlabel=c,ylabel='Training rows')
    save_figure('F50',fig)
    fig,axes=axes_grid(len(derived.columns))
    for ax,c in zip(axes,derived):
        ax.hist(derived[c].dropna(),bins=40)
        ax.set(xlabel=c,ylabel='Training rows')
    save_figure('F51',fig)
    transform_corr=pd.DataFrame({'Post GPA':derived.corrwith(train[TARGET_COLUMN]),
        'GPA Change':derived.corrwith(train.GPA_Change)})
    fig,ax=plt.subplots(figsize=(8,max(4,.5*len(transform_corr))))
    sns.heatmap(transform_corr,vmin=-1,vmax=1,center=0,cmap='vlag',annot=True,fmt='.3f',ax=ax)
    save_figure('F52',fig,caption='Marginal training correlations, not evidence of incremental forecast benefit.')
    pair=train[[H,TARGET_COLUMN]].dropna()
    if pair[H].nunique()>=3:
        fig,ax=plt.subplots(figsize=(8,5))
        shown_pair=plot_sample(pair)
        ax.scatter(shown_pair[H],shown_pair[TARGET_COLUMN],s=5,alpha=.1)
        grid=np.linspace(pair[H].min(),pair[H].max(),150)
        for degree in [1,2]:
            coefs=np.polyfit(pair[H],pair[TARGET_COLUMN],degree)
            ax.plot(grid,np.polyval(coefs,grid),label=f'Marginal degree {degree} (train)')
        trend=binned_trend(train,H,TARGET_COLUMN)
        ax.plot(trend.x_mean,trend.y_mean,'o',color='black',label='Train decile mean')
        ax.legend()
        ax.set(xlabel='Weekly GenAI hours/week',ylabel='Post GPA (points)')
        save_figure('F53',fig,caption='This figure is marginal. T35 quadratic validation comparison adjusts using fixed E3 features.')
    else:
        skip('F53','AI Hours has fewer than 3 distinct values for quadratic comparison')
    before=train[[c for c in [H,'Traditional_Study_Hours','Pre_Semester_GPA'] if c in train]].corr()
    after=pd.concat([train[[c for c in [H,'Traditional_Study_Hours','Pre_Semester_GPA'] if c in train]],derived[['Log_AI_Hours','Centered_AI_Hours_Squared']]],axis=1).corr()
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,matrix,label in [(axes[0],before,'Original predictor correlation'),(axes[1],after,'Candidate engineering correlation')]:
        sns.heatmap(matrix,vmin=-1,vmax=1,center=0,cmap='vlag',annot=True,fmt='.2f',ax=ax)
        ax.set_title(label)
    save_figure('F54',fig,caption='Audit candidates together; production design must avoid raw H + centered H or total + components.')
else:
    for aid in ['F50','F51','F52','F53','F54']:
        skip(aid,'AI Hours unavailable; no derived feature analysis')
note('**Recommendation.** T35 phân biệt giảm skewness với cải thiện validation. Centering có thể giảm tương quan H–H², '
     'nhưng không thay thế kiểm tra VIF toàn design. Total/ratio và numeric ordinal products chưa được tự ý thêm. '
     'Polynomial/interaction giữ main effects; Ridge cần scale và xác minh được môn học chấp nhận.')
""")
md(r"""
## Section 11 — Data Leakage & Bias Analysis

**Method.** Feature availability là guard thực thi từ trước split/model fit. Chỉ Pre GPA có temporal evidence rõ từ mô tả;
AI/study/policy/demographic còn lại chưa có timestamp nên conditional, anxiety có temporal risk cao.
Test audit dùng assignment/group keys và full-file structural hashes, không profile target test.
Synthetic status/license được xác nhận từ metadata chính thức; không suy thêm sampling design hoặc cơ chế sinh dữ liệu.
""")
code(r"""
availability_rows=[]
leakage_rows=[]
for c in raw:
    forbidden=c in [ID_COLUMN,TARGET_COLUMN,*auxiliary] or c in extra_columns
    availability='unavailable' if forbidden else 'available' if c in VERIFIED_AVAILABLE else 'unknown'
    risk='high' if forbidden or c=='Anxiety_Level_During_Exams' else 'low' if c in VERIFIED_AVAILABLE else 'conditional'
    availability_rows.append(dict(feature=c,availability=availability,prediction_cutoff=PREDICTION_CUTOFF,
        source_evidence='GPA at start of semester' if c in VERIFIED_AVAILABLE else 'post-semester outcome / identifier / unknown role' if forbidden else 'measurement timing not specified by publisher',
        allowed_scenario='not a predictor' if forbidden else 'pre-semester baseline' if c in VERIFIED_AVAILABLE else 'conditional / retrospective only',
        forecast_eligible=bool(c in VERIFIED_AVAILABLE and not forbidden),leakage_risk=risk))
    leakage_rows.append(dict(feature=c,issue='role/target leakage' if forbidden else 'temporal availability',risk=risk,
        action='Exclude from X' if forbidden else 'Allow baseline' if c in VERIFIED_AVAILABLE else 'Conditional only; verify timestamp before forecasting',
        evidence='source metadata; T02; T33; code model allowlist'))
leakage_rows.append(dict(feature='GPA_Change',issue='target-derived leakage',risk='high',action='Analytical target only; never X',evidence='Post minus Pre formula'))
save_table('T39',leakage_rows,'split_metadata')
availability_table=save_table('T40',availability_rows,'split_metadata')
split_checks=[]
split_labels=['train','validation','test']
for i,left in enumerate(split_labels):
    li=assignments[assignments.split.eq(left)]
    split_checks.append(dict(check='split row count',left=left,right='',value=len(li),passed=len(li)>0,scope='split_metadata'))
    for right in split_labels[i+1:]:
        ri=assignments[assignments.split.eq(right)]
        row_overlap=len(set(li.row_key)&set(ri.row_key))
        group_overlap=len(set(li.group_key)&set(ri.group_key))
        content_overlap=len(set(content_hash.loc[li.source_row_position])&set(content_hash.loc[ri.source_row_position]))
        for name,value in [('row-key overlap',row_overlap),('connected-group overlap',group_overlap),('content-hash overlap',content_overlap)]:
            split_checks.append(dict(check=name,left=left,right=right,value=value,passed=value==0,scope='split_metadata'))
        if ID_COLUMN in clean:
            overlap=len(set(clean.loc[li.source_row_position,ID_COLUMN].dropna()) & set(clean.loc[ri.source_row_position,ID_COLUMN].dropna()))
            split_checks.append(dict(check='Student ID overlap',left=left,right=right,value=overlap,passed=overlap==0,scope='split_metadata'))
split_checks.extend([
    dict(check='all source rows assigned once',left='all',right='',value=len(assignments),passed=len(assignments)==len(raw) and assignments.row_key.is_unique,scope='split_metadata'),
    dict(check='raw CSV unchanged',left='raw',right='',value=raw_fingerprint,passed=digest_file(csv_path)==raw_fingerprint,scope='full_structural_audit'),
    dict(check='forbidden predictors absent',left='models',right='',value=len(models),passed=all(not set(m['features'])&{ID_COLUMN,TARGET_COLUMN,'GPA_Change',*auxiliary} for m in models.values()),scope='split_metadata'),
    dict(check='no test target/model frame materialized',left='test',right='',value='assignment/count/integrity only',passed='test' not in globals() and 'y_test' not in globals() and 'X_test' not in globals(),scope='split_metadata')])
integrity=save_table('T41',split_checks,'split_metadata')
assert integrity.passed.all(), 'Split/role/raw-file integrity failed; do not use outputs.'
limitations=[
 dict(issue='synthetic provenance',status=source_meta.get('provenance','unverified'),evidence='official user_specified_sources in saved metadata',
      consequence='Learns simulated patterns; external validity to real students unverified',action='No causal or population claims'),
 dict(issue='license and version',status=f"{source_meta.get('license','unverified')}; publisher version={source_meta.get('dataset_version','unverified')}",evidence=SOURCE_URL,
      consequence='Local bytes not authenticated to publisher checksum',action='Record local SHA-256; retain source metadata snapshot'),
 dict(issue='sampling / independence',status='unknown',evidence='publisher does not describe sampling or generation mechanism',
      consequence='SE/CI are descriptive; representative inference unsupported',action='Audit real cohort before external use'),
 dict(issue='feature timing',status='mostly unknown',evidence='T40; source definitions',consequence='E2/E3 not verified forecasts',action='Obtain measurement dates / freeze cutoff'),
 dict(issue='hours coverage',status='study additivity unverified',evidence='AI usage hours not explicitly restricted to study',consequence='Total/ratio may misstate study time',action='Skip unless documented'),
 dict(issue='GPA boundaries / precision',status='upper documented; lower and precision unverified',evidence='T18; source approximate ranges',consequence='Floor effect unknown; equality tolerance only numerical',action='No inferred floor; no silent prediction clipping'),
 dict(issue='group imbalance',status=f'{int(frequencies.small_group.sum()) if len(frequencies) else 0} rare category rows in frequency table',evidence='T12; T21; T24',
      consequence='Small subgroups have unstable summaries',action=f'Flag n<{MIN_GROUP_SIZE}; report counts'),
 dict(issue='holdout policy',status='test untouched for analysis',evidence='T41; split assignments',consequence='Validation is exploratory, not final estimate',action='Final test only after model specification frozen'),
 dict(issue='course source PDFs',status='unavailable in workspace',evidence='brief.md references two PDF files',consequence='Exact original rubric not verified',action='Confirm final model count versus team size and Ridge allowance')]
save_table('T42',limitations,'split_metadata')
fig,ax=plt.subplots(figsize=(7,4))
counts=assignments.split.value_counts().reindex(['train','validation','test','excluded'],fill_value=0)
ax.bar(counts.index,counts.values)
for i,value in enumerate(counts):
    ax.text(i,value,str(value),ha='center',va='bottom')
ax.set(xlabel='Frozen split assignment',ylabel='Source rows (no target summaries)')
save_figure('F55',fig,'split_metadata')
if PLOT_VALIDATION_TARGET:
    fig,ax=plt.subplots(figsize=(8,4))
    bins=np.histogram_bin_edges(train[TARGET_COLUMN],bins=40)
    ax.hist(train[TARGET_COLUMN],bins=bins,density=True,alpha=.5,label='train')
    ax.hist(validation[TARGET_COLUMN],bins=bins,density=True,alpha=.5,label='validation')
    ax.set(xlabel='Post GPA (points)',ylabel='Density')
    ax.legend()
    save_figure('F56',fig,'validation','Optional audit only; histogram bin edges fitted to train. No test targets.')
else:
    skip('F56','Optional train-validation target profiling disabled by configuration; test remains untouched','validation')
fig,ax=plt.subplots(figsize=(8,max(5,.35*len(availability_table))))
risk_code=availability_table.leakage_risk.map({'low':0,'conditional':1,'high':2})
matrix=pd.DataFrame({'Verified available before cutoff':availability_table.forecast_eligible.astype(int),
                    'Leakage risk (0 low, 1 conditional, 2 high)':risk_code},index=availability_table.index)
matrix.index=availability_table.feature
sns.heatmap(matrix,annot=True,fmt='d',cmap='YlOrRd',vmin=0,vmax=2,cbar_kws={'label':'Column-specific encoded status; see labels'},ax=ax)
save_figure('F57',fig,'split_metadata',caption='Administrative status matrix, not a fitted feature importance or statistical score. Source evidence in T40.')
note('**Implication.** Split/role guards đều được kiểm tra bằng assertion ở T41. '
     'Synthetic data và timing unknown là giới hạn hiện hữu; dù validation score tốt vẫn chưa chứng minh AI gây GPA thay đổi hay mô hình dùng được trên cohort thật.')
""")
md(r"""
## Section 12 — EDA Findings & Recommendations

**Method.** Mỗi finding có ID, RQ, evidence IDs, actual values, scope, interpretation, action và limitation.
Không thay missing evidence bằng số tự tạo. Feature decisions kết hợp semantic role, availability, train missingness,
correlations và **term-level VIF**, không gán feature importance tổng hợp tùy ý.
Khuyến nghị M01–M05 là thiết kế cho bước training tiếp theo; số model cuối cùng ít nhất bằng số thành viên thật.
Ridge cần xác nhận phạm vi phương pháp được giảng viên chấp nhận; notebook không tuning model cuối hay đánh giá test.
""")
code(r"""
findings=[]
def finding(rq,observation,scope,evidence,interpretation,action,limitation,status='resolved'):
    findings.append(dict(finding_id=f'FND-{len(findings)+1:03}',research_question=rq,status=status,
        observation=observation,data_scope=scope,evidence_ids=evidence,
        interpretation=interpretation,recommended_action=action,limitation=limitation))
def corr_sentence(feature,outcome=TARGET_COLUMN,method='spearman'):
    table=spearman_table if method=='spearman' else pearson_table
    part=table[table.feature.eq(feature)&table.outcome.eq(outcome)]
    if part.empty:
        return f'{feature} vs {outcome}: unresolved (missing applicable feature)'
    r=part.iloc[0]
    return f'{feature} vs {outcome}: {method}={r.coefficient:.6g}, valid n={int(r.valid_n)}'+(f'; {r.reason}' if r.reason else '')

finding('RQ01',f'Raw rows={len(raw)}, columns={len(raw.columns)}; raw blank/missing cells={int(missing_table.missing_count.sum())}; '
    f'exact duplicates removed={int(exact_mask.sum())}; eligible={len(eligible_idx)}; logged invalid checks affected rows (summed, may overlap)={int(pd.DataFrame(invalid_log).affected_rows.sum())}',
    'full_structural_audit','T01 T05 T06 T07 T09 T41','Audit/cleaning có quy tắc, không impute target.',
    'Giữ raw read-only và frozen splits; fit preprocessing train only.','Completeness không bảo đảm dữ liệu synthetic đại diện thực tế.')
e1=results.set_index('model').loc['E1'] if 'E1' in set(results.model) else None
finding('RQ02',f'E1 validation n={int(e1.valid_n)}, MAE={e1.MAE:.6g}, RMSE={e1.RMSE:.6g}, R2={e1.R2:.6g}; '
    f'E0a RMSE={results.set_index("model").loc["E0a","RMSE"]:.6g}' if e1 is not None else 'E1 unavailable: no variable Pre GPA',
    'validation','T15 T29 F16 F49','Pre GPA là baseline trước kỳ; mức dự đoán được lượng hóa trên holdout validation.',
    'Giữ M01 và mean/persistence reference trong training.','Validation exploratory; chưa phải final test.',status='resolved' if e1 is not None else 'unresolved')
finding('RQ03',corr_sentence(H)+'; '+corr_sentence(H,'GPA_Change'),'train','T19 T20 F24 F32',
    'Liên hệ marginal khác với giá trị dự đoán có điều kiện.','Kiểm chứng H và quadratic candidate cùng academic controls.',
    'Timing unknown; GPA Change bị coupling với Pre GPA.',status='resolved' if H in train else 'unresolved')
finding('RQ04',corr_sentence('Traditional_Study_Hours')+'; '+corr_sentence('Traditional_Study_Hours','GPA_Change'),
    'train','T19 T20 F35','Weekly study là candidate trong retrospective academic model.',
    'So M02 với M01, xác minh measurement window trước forecast.','Không mặc định GenAI usage cộng được với study.',
    status='resolved' if 'Traditional_Study_Hours' in train else 'unresolved')
group_evidence='; '.join(f'{r["feature"]}: groups={r["groups"]}, range of group mean={r["mean_range"]:.6g}, small groups={r["small_groups"]}'
    for r in group_differences if r['outcome']==TARGET_COLUMN and r['feature'] in ['Prompt_Engineering_Skill','Primary_Use_Case','Major_Category'])
finding('RQ05',group_evidence or 'No applicable subgroup evidence','train','T21 T22 F25 F27 F28',
    'Group differences được báo cùng size và uncertainty.','One-hot; flag rare groups; giữ main effects khi kiểm tra interaction.',
    'Unadjusted differences; multiple comparisons chưa được dùng làm inferential claims.',status='resolved' if group_evidence else 'unresolved')
dep_ret=[r for r in aux_records if r.get('feature')=='Perceived_AI_Dependency' and r.get('outcome')==retention]
finding('RQ06',corr_sentence('Perceived_AI_Dependency')+'; '+corr_sentence('Perceived_AI_Dependency','GPA_Change')+
    ('; retention Spearman='+f'{dep_ret[0]["coefficient"]:.6g}'+f', n={dep_ret[0]["valid_n"]}' if dep_ret else '; retention unresolved'),
    'train','T20 T26 F37 F39','Dependency là ordinal, timing chưa rõ; retention là auxiliary outcome.',
    'Dùng rank để mô tả; one-hot cho conditional model; retention tuyệt đối không vào X.','Không coi score ordinal là interval scale tự động.',
    status='resolved' if 'Perceived_AI_Dependency' in train else 'unresolved')
quad=[r for r in transformation_comparison if r['candidate']=='Centered_AI_Hours_Squared']
base_r=results.set_index('model').loc['E3'] if 'E3' in set(results.model) else None
quad_valid=bool(quad and np.isfinite(quad[0]['validation_RMSE']))
quad_observation=(f'Fixed quadratic validation RMSE={quad[0]["validation_RMSE"]:.6g}; E3 RMSE={base_r.RMSE:.6g}; '
    f'delta={quad[0]["validation_RMSE"]-base_r.RMSE:.6g}; coefficient={quad[0]["quadratic_coefficient"]:.6g}' if quad_valid and base_r is not None else 'Quadratic comparison unresolved; inspect T33/T35')
finding('RQ07',quad_observation+f'; cross-tab cells={len(interaction_summaries)}; grouped-trend cells={len(multivariate_rows)}',
    'train + limited validation','T24 T25 T26 T33 T35 F24 F53','Observed curvature là hypothesis cho specification; interaction grids là descriptive.',
    'M04 giữ H khi thêm centered H²; kiểm chứng interaction có lower-order terms.',
    'Một validation comparison định trước, chưa tuning degree hoặc xác nhận causal moderation.',status='resolved' if quad_valid else 'unresolved')
if 'E3' in set(results.model) and 'E2' in set(results.model):
    e2,e3=results.set_index('model').loc[['E2','E3']].to_dict('index').values()
    ai_delta=f'E3 minus E2: delta MAE={e3["MAE"]-e2["MAE"]:.6g}, delta RMSE={e3["RMSE"]-e2["RMSE"]:.6g}, delta R2={e3["R2"]-e2["R2"]:.6g}; validation n={int(e3["valid_n"])}'
else:
    ai_delta='E2/E3 comparison unavailable'
finding('RQ08',ai_delta,'validation','T29 F49','Negative delta error / positive delta R² means incremental association-based predictive benefit within this scenario.',
    'Giữ cùng rows/metrics và ghi rõ M03 là conditional khi timing chưa xác minh.','Không gọi validation là test; exam anxiety có temporal risk cao.',
    status='resolved' if 'E3' in models and 'E2' in models else 'unresolved')
finding('RQ09',f'{diagnostic_id} rank={diagnostic_model["rank"]}/{p}; encoded terms={len(vif_table)}; '
    f'infinite VIF terms={int(np.isinf(vif_table.VIF).sum())}; max finite VIF={vif_table.VIF.replace(np.inf,np.nan).max():.6g}; '
    f'flagged quantitative high-correlation pairs={len(high_pairs)}',
    'train','T27 T28 F41 F54','Rank/VIF xét predictor design có reference category và intercept.',
    'Không tự động xóa theo heuristic; dùng Ridge candidate nếu coefficients thiếu ổn định.',
    'Term-level VIF phụ thuộc encoding/feature set; raw condition number phụ thuộc đơn vị.')
finding('RQ10',f'{len(encoding_rows)} observed features have encoding plans; {len(derived.columns)} derived train candidates; '
    f'hours additivity verified={CONFIRMED_HOURS_ADDITIVE}',
    'train + metadata','T33 T34 T35 T36 T37 T38','Impute/one-hot train-only; transformations dựa domain và hierarchy.',
    'Median numeric, mode + one-hot categorical; scale polynomial/Ridge trong train pipeline.',
    'No forced total/ratio; no equal-spacing assumption for ordinal numeric interactions.')
finding('RQ11',f'Train residual skew={pd.Series(residual).skew():.6g}; abs standardized residual >3: {int((np.abs(standardized)>3).sum())}; '
    f'Cook D >4/n: {int((cooks>4/n).sum())}; leverage >2p/n: {int((leverage>2*p/n).sum())}; n={n}, p={p}',
    'train','T18 T31 T32 F43 F45 F46 F48','Residual diagnostics có thể chỉ ra giới hạn linear mean, variance và bounded target.',
    'Inspect residual structure; cân nhắc HC3, polynomial candidate hoặc regularization; không tự xóa flags.',
    'Independence không xác minh; normality cần cho inference, không phải yêu cầu predictors chuẩn.')
finding('RQ12',f'Publisher provenance={source_meta.get("provenance","unverified")}; license={source_meta.get("license","unverified")}; '
    f'version={source_meta.get("dataset_version","unverified")}; verified forecast features={int(availability_table.forecast_eligible.sum())}',
    'metadata','T40 T42 T48','Synthetic patterns không tự khái quát đến sinh viên thật.',
    'Cần cohort validation và measurement timestamps trước triển khai.',
    'Generator, representativeness, true GPA floor/precision, course PDFs và team size chưa xác minh.')

rq_evidence=save_table('T43',pd.DataFrame(findings)[['research_question','finding_id','status','evidence_ids','data_scope']])
findings_table=save_table('T44',findings,caption='All numeric observations are generated from current run; unresolved evidence stays explicitly unresolved.')
decision_rows=[]
for row in selection_rows:
    c=row['feature']
    avail=availability_table.set_index('feature').loc[c]
    pe=pearson_table[pearson_table.feature.eq(c)&pearson_table.outcome.eq(TARGET_COLUMN)]
    sp=spearman_table[spearman_table.feature.eq(c)&spearman_table.outcome.eq(TARGET_COLUMN)]
    terms=vif_table[vif_table.original_feature.eq(c)][['encoded_term','VIF']]
    decision_rows.append(dict(feature=c,semantic_type=semantic.get(c,'unknown'),availability=avail.availability,
        train_missing_pct=100*train[c].isna().mean(),pearson=pe.coefficient.iloc[0] if len(pe) else np.nan,
        spearman=sp.coefficient.iloc[0] if len(sp) else np.nan,
        encoded_VIF_terms=json.dumps([{ 'term':r.encoded_term, 'VIF':str(r.VIF)} for r in terms.itertuples()]),
        leakage_risk=avail.leakage_risk,encoding=next(r['encoding'] for r in encoding_rows if r['feature']==c),
        transformation='centered H²/log candidates; hierarchy required' if c==H else 'none committed',
        decision=row['decision'],reason=row['rationale'],evidence_ids='T02 T19 T20 T28 T33 T35 T40'))
decision_matrix=save_table('T45',decision_rows)
plan_rows=[]
for row in decision_rows:
    plan_rows.append(dict(feature_or_issue=row['feature'],action=row['encoding'],decision=row['decision'],
        rationale=row['reason'],fit_scope='train only',EDA_executed='minimal cleaning and exploratory preprocessing only',
        training_proposal='freeze pipeline, verify timing, preserve test holdout',evidence_ids=row['evidence_ids']))
plan_rows.extend([
 dict(feature_or_issue='target missing/invalid',action='exclude row; no imputation',decision='Exclude',rationale='supervised target validity',
      fit_scope='structural rules before split',EDA_executed=f'{int((exclusion=="missing_or_invalid_target").sum())} rows excluded',training_proposal='same protocol',evidence_ids='T07 T09'),
 dict(feature_or_issue='valid outliers / influence',action='retain; investigate',decision='Keep',rationale='heuristics do not prove error',
      fit_scope='thresholds train only',EDA_executed='no automatic deletion',training_proposal='optional train sensitivity analysis, no test-driven removal',evidence_ids='T08 T31 T32'),
 dict(feature_or_issue='polynomial / regularization',action='train-centered H² + main effect; scale for Ridge',decision='Investigate',rationale='T35 fixed comparison; VIF/rank diagnostics',
      fit_scope='train/CV; validation exploratory',EDA_executed='one fixed quadratic extension, no tuning',training_proposal='freeze model before final test',evidence_ids='T28 T33 T35'),
 dict(feature_or_issue='Total hours / AI study ratio',action='skip unless additivity independently verified',decision='Investigate',rationale='AI usage not necessarily study',
      fit_scope='documented semantic rule',EDA_executed='skipped by default',training_proposal='denominator=0 -> missing if later enabled',evidence_ids='T33 T42')])
preprocessing_plan=save_table('T46',plan_rows)
model_recommendations=[
 dict(model='M01',method='Simple Linear Regression',features='Pre_Semester_GPA',scenario='pre-semester baseline',
      rationale='E1 vs E0a/E0b in T29',evidence_ids='T15 T29',status='training candidate; no final test evaluation'),
 dict(model='M02',method='Multiple Linear Regression Academic',features='; '.join(academic),scenario='conditional / retrospective unless all timings verified',
      rationale='Academic increment measured E2 vs E1; scenario differs',evidence_ids='T19 T21 T29 T40',status='training candidate'),
 dict(model='M03',method='Multiple Linear Regression Academic + AI',features='; '.join(dict.fromkeys(academic+ai_additions)),scenario='conditional / retrospective; anxiety high temporal risk',
      rationale=ai_delta,evidence_ids='T28 T29 T40',status='training candidate; verify timing or exclude anxiety for deployment'),
 dict(model='M04',method='Polynomial Linear Regression',features='M03 + centered H^2; selected reference-coded interactions only with main effects',scenario='conditional / retrospective',
      rationale=quad_observation,evidence_ids='T24 T28 T33 T35',status='training candidate; degree/interaction selection via train CV'),
 dict(model='M05',method='Ridge Regression',features='same eligible encoded feature set as chosen M02/M03 scenario; scale within train pipeline',scenario='match selected feature set and availability',
      rationale='test coefficient stability against OLS based on rank/VIF; confirm course allowance',evidence_ids='T28 T37 T40',status='proposal only; no tuning in this EDA')]
save_table('T47',model_recommendations)
save_table('T48',limitations,'split_metadata')
note('**Findings and actions.**')
for r in findings:
    note(f'**{r["finding_id"]} / {r["research_question"]} [{r["data_scope"]}]** — {r["observation"]}  \n'
         f'{r["interpretation"]} **Action:** {r["recommended_action"]}  \n'
         f'*Limitation:* {r["limitation"]} Evidence: {r["evidence_ids"]}.')
""")
md(r"""
### Export and run assertions

Xuất reports UTF-8 và manifest đủ 48 tables/57 figures. Mọi file generated/reused phải tồn tại và đọc được;
optional/không áp dụng phải ghi skipped với reason. Các assert bên dưới không thay cho kiểm chứng notebook thực thi bằng kernel sạch;
script `scripts/execute_notebook.py --verify-rerun` chạy hai lần và báo AC01–AC18.
""")
code(r"""
pending=[aid for aid,r in manifest.items() if r['status']=='pending']
assert not pending, f'Unaccounted artifacts: {pending}'
manifest_frame=pd.DataFrame(manifest.values()).sort_values('artifact_id').reset_index(drop=True)
assert len(manifest_frame)==105 and manifest_frame.artifact_id.is_unique
for r in manifest_frame.itertuples():
    if r.status in ['generated','reused']:
        path=OUTPUT_DIR/r.relative_path
        assert path.is_file() and path.stat().st_size>0, f'{r.artifact_id}: missing artifact'
        if r.artifact_type=='table':
            pd.read_csv(path)
        else:
            plt.imread(path)
    else:
        assert r.status=='skipped' and r.reason and not r.relative_path
manifest_frame.to_csv(OUTPUT_DIR/'artifact_manifest.csv',index=False,encoding='utf-8')
pd.DataFrame(preprocessing_log).to_csv(OUTPUT_DIR/'preprocessing_log.csv',index=False,encoding='utf-8')

def artifact_links(ids):
    links=[]
    for aid in ids.split():
        r=manifest[aid]
        links.append(f'[{aid}]({r["relative_path"]})' if r['relative_path'] else f'{aid} (skipped: {r["reason"]})')
    return ', '.join(links)

report_lines=['# EDA Findings', '', f'Dataset SHA-256: `{raw_fingerprint}`',
    f'CSV: `{csv_path.name}` · raw rows: {len(raw):,} · columns: {len(raw.columns)}',
    f'Split: train={len(train_idx):,}; validation={len(val_idx):,}; test={len(test_idx):,}; excluded={int(exclusion.ne("").sum()):,}.',
    'Scope: detailed EDA on train, prespecified comparisons on validation; test counts/integrity only.',
    'Source: ['+SOURCE_URL+']('+SOURCE_URL+'). Saved metadata: provenance='+str(source_meta.get('provenance','unverified'))+
    ', license='+str(source_meta.get('license','unverified'))+', publisher version='+str(source_meta.get('dataset_version','unverified'))+'. Local byte-version match unverified.',
    '', '## Research questions', '', rq_evidence.to_markdown(index=False), '', '## Findings', '']
for r in findings:
    report_lines.extend([f'### {r["finding_id"]} — {r["research_question"]}', '',
        f'**Observation ({r["data_scope"]}; {r["status"]}):** {r["observation"]}', '',
        r['interpretation'], '', '**Action:** '+r['recommended_action'], '',
        '**Limitation:** '+r['limitation'], '', '**Evidence:** '+artifact_links(r['evidence_ids']), ''])
report_lines.extend(['## Model candidates', '', pd.DataFrame(model_recommendations).to_markdown(index=False), '',
    'MAE, MSE, RMSE, R² use the same target/splits/population. Final model count must be at least team size (unknown).', '',
    '## Unresolved and limitations', '', pd.DataFrame(limitations).to_markdown(index=False), '',
    'No causal AI impact claim; no population generalization from synthetic data; no final test evaluation.'])
(OUTPUT_DIR/'EDA_FINDINGS.md').write_text('\n'.join(report_lines),encoding='utf-8')
(OUTPUT_DIR/'DATA_DICTIONARY.md').write_text('# Data Dictionary\n\n'
    'Observed min/max/categories/unique: TRAIN only. Structural missing/schema: full file. Identifier values withheld.\n\n'
    +f'SHA-256: `{raw_fingerprint}`\n\n'+tables['T02'].to_markdown(index=False)+'\n\n'
    +'Units/domains/timing from publisher metadata where explicit; unknowns are retained. Source: ['+SOURCE_URL+']('+SOURCE_URL+').\n',encoding='utf-8')
(OUTPUT_DIR/'PREPROCESSING_PLAN.md').write_text('# Preprocessing Plan\n\n'
    'EDA has performed deterministic minimal cleaning and exploratory train-only preprocessing. Training proposals below are not final fitted pipelines.\n\n'
    +preprocessing_plan.to_markdown(index=False)+'\n\n## Feature decisions\n\n'+decision_matrix.to_markdown(index=False)
    +'\n\n## Engineering definitions\n\n'+tables['T33'].to_markdown(index=False)
    +'\n\n## Model candidates\n\n'+tables['T47'].to_markdown(index=False),encoding='utf-8')

assert digest_file(csv_path)==raw_fingerprint, 'Raw dataset changed during execution'
run_metadata.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat(),
    split_metadata=split_metadata, artifact_status_counts=manifest_frame.status.value_counts().to_dict(),
    artifact_coverage={'tables':48,'figures':57}, warnings=warnings_log,
    unresolved=['local CSV publisher checksum/version authentication','sampling/generation mechanism',
        'most feature timestamps','true GPA lower bound and precision','AI hours study coverage',
        'course PDFs and actual team size','external validation on real students'],
    invariant_checks={'raw_unchanged':True,'all_splits_disjoint':True,'no_forbidden_predictors':True,'manifest_paths_readable':True,
        'test_policy':'counts/integrity only; no target profile, metrics or fitted preprocessing'},
    report_files=['EDA_FINDINGS.md','DATA_DICTIONARY.md','PREPROCESSING_PLAN.md','artifact_manifest.csv','preprocessing_log.csv'])
json_write(OUTPUT_DIR/'run_metadata.json',run_metadata)
note(f'**Run completed.** {len(manifest_frame)} planned IDs: '+str(manifest_frame.status.value_counts().to_dict())+
     f'. Reports: `{OUTPUT_DIR}`. Splits: `{SPLIT_DIR}`. Raw SHA-256 unchanged: `{raw_fingerprint}`. '
     'Đọc EDA_FINDINGS.md để xem RQ → evidence → interpretation → modeling action.')
display(manifest_frame.groupby(['artifact_type','status']).size().rename('count').reset_index())
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.13'},
    'description': 'Self-contained reproducible EDA, train-only detailed analysis, frozen holdout, honest artifact manifest.'})
nbf.validate(notebook)
for i, cell in enumerate(notebook.cells):
    if cell.cell_type == 'code':
        compile(cell.source, f'cell_{i}', 'exec')
target = ROOT / 'notebooks' / 'kaggle' / '01_eda.ipynb'
target.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, target)
print(f'Built {target}: {len(cells)} cells, {sum(c.cell_type=="code" for c in cells)} executable cells')
