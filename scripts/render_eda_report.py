"""Render a readable Vietnamese EDA report from the current executed notebook artifacts.

No model fitting or data profiling is performed here. Selected figures are embedded in HTML.
"""
import argparse
import base64
from datetime import datetime, timedelta, timezone
from html import escape
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CSS = """
:root{--ink:#172c3e;--muted:#536779;--accent:#087e83;--line:#dce5ea;--paper:#fff;--soft:#eef6f6}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#edf1f4;color:var(--ink);font:16px/1.65 'Segoe UI',Arial,sans-serif}
main{max-width:1200px;margin:32px auto;background:var(--paper);padding:54px 60px;box-shadow:0 6px 30px #12273510}
header{border-top:6px solid var(--accent);padding-top:24px;margin-bottom:32px}h1{font-size:38px;line-height:1.15;letter-spacing:-1px;margin:10px 0 18px}h2{font-size:25px;border-bottom:1px solid var(--line);padding-bottom:10px;margin-top:52px;scroll-margin-top:22px}h3{font-size:18px;margin-bottom:8px}p{margin:12px 0}a{color:#086c85;text-underline-offset:3px}small,.meta,figcaption{color:var(--muted)}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:12px;font-weight:700;color:var(--accent)}.meta{font-size:13px;overflow-wrap:anywhere}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.stat{border:1px solid var(--line);padding:18px;border-radius:5px}.stat strong{display:block;font-size:27px;color:var(--accent)}.stat span{font-size:13px;color:var(--muted)}.callout{background:var(--soft);border-left:4px solid var(--accent);padding:14px 18px;margin:20px 0}.toc{columns:2;padding:18px 30px;background:#f7f9fa;border:1px solid var(--line)}.toc a{display:block;padding:4px 0;font-size:14px;text-decoration:none}.table-wrap{overflow-x:auto;margin:16px 0 24px}table{width:100%;border-collapse:collapse;font-size:12px;line-height:1.5}th,td{padding:9px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line);max-width:450px;overflow-wrap:anywhere}th{background:#edf4f6;color:#29495b;font-weight:650}tbody tr:nth-child(even){background:#fafcfd}td:empty::after{content:'—';color:#93a1ac}figure{margin:26px 0;padding:12px;border:1px solid var(--line);border-radius:5px;background:white}figure img{display:block;width:100%;height:auto}figcaption{font-size:13px;padding:8px 7px}.finding{border-bottom:1px solid var(--line);padding:12px 0 20px}.finding h3{color:var(--accent)}.finding .meta{font-size:12px}.evidence{font-size:13px}.print{float:right;border:1px solid var(--line);background:white;border-radius:4px;padding:8px 14px;color:var(--muted);cursor:pointer}footer{margin-top:42px;border-top:2px solid var(--line);padding-top:18px;font-size:12px;color:var(--muted)}code{font-size:12px;overflow-wrap:anywhere}
@media(max-width:760px){main{margin:0;padding:26px 18px}h1{font-size:30px}.stats{grid-template-columns:repeat(2,1fr)}.toc{columns:1}.print{display:none}}
@media print{body{background:white;font-size:10pt}main{margin:0;padding:0;box-shadow:none;max-width:none}.print,.toc{display:none}h2{break-after:avoid}figure{break-inside:avoid}a{color:inherit}table{font-size:8pt}.stats{grid-template-columns:repeat(4,1fr)}@page{size:A4;margin:15mm}}
"""


def render(output_dir):
    meta_path = output_dir / 'run_metadata.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if meta.get('status') != 'completed':
        raise ValueError('Execute the notebook successfully before rendering the report.')
    manifest = pd.read_csv(output_dir / 'artifact_manifest.csv').fillna('').set_index('artifact_id')
    verification = json.loads((output_dir / 'verification_results.json').read_text(encoding='utf-8'))
    if verification['dataset_sha256'] != meta['dataset_sha256']:
        raise ValueError('Verification and current dataset fingerprint differ; rerun execution verification.')
    tables = {}
    for aid, row in manifest.iterrows():
        if row.artifact_type == 'table' and row.status in ['generated', 'reused']:
            tables[aid] = pd.read_csv(output_dir / row.relative_path)

    def t(aid):
        return tables[aid].copy()

    html = []
    markdown = []
    used = set()

    def paragraph(text, alert=False):
        html.append(f'<div class="callout">{escape(text)}</div>' if alert else f'<p>{escape(text)}</p>')
        markdown.extend([('> ' if alert else '') + text, ''])

    def section(number, title):
        html.append(f'<section id="s{number}"><h2>{number:02}. {escape(title)}</h2>')
        markdown.extend([f'## {number:02}. {title}', ''])

    def end_section():
        html.append('</section>')

    def table(aid, columns=None, subset=None, title=None):
        if aid not in tables:
            paragraph(f'{aid}: skipped — {manifest.loc[aid,"reason"]}')
            return
        frame = t(aid) if subset is None else subset.copy()
        if columns:
            frame = frame[[c for c in columns if c in frame.columns]]
        label = f'{aid} · {title or manifest.loc[aid,"title"]}'
        link = manifest.loc[aid, 'relative_path']
        html.append(f'<h3>{escape(label)}</h3><p class="meta">Scope: {escape(manifest.loc[aid,"data_scope"])} · <a href="{escape(link)}">CSV đầy đủ</a></p>')
        display_frame = frame.replace([np.inf, -np.inf], ['∞', '−∞'])
        html.append('<div class="table-wrap">' + display_frame.to_html(index=False, border=0, na_rep='—',
            float_format=lambda v: f'{v:,.5g}', escape=True) + '</div>')
        markdown.extend([f'### {label}', '', f'Scope: `{manifest.loc[aid,"data_scope"]}` · [CSV đầy đủ]({link})', '',
                         display_frame.to_markdown(index=False, floatfmt='.5g', missingval='—'), ''])
        used.add(aid)

    def figure(aid, caption):
        row = manifest.loc[aid]
        if row.status not in ['generated', 'reused']:
            paragraph(f'{aid}: skipped — {row.reason}')
            return
        path = output_dir / row.relative_path
        blob = base64.b64encode(path.read_bytes()).decode('ascii')
        label = f'{aid} · {row.title}'
        html.append(f'<figure><img src="data:image/png;base64,{blob}" alt="{escape(label)}" loading="lazy">'
                    f'<figcaption><strong>{escape(label)}</strong> · {escape(row.data_scope)}. {escape(caption)}</figcaption></figure>')
        markdown.extend([f'![{label}]({row.relative_path})', '', caption, ''])
        used.add(aid)

    title = 'EDA Report — AI Impact on Students'
    run_time = datetime.fromisoformat(meta['completed_at']).astimezone(timezone(timedelta(hours=7)))
    train_n = meta['split_metadata']['counts']['train']
    val_n = meta['split_metadata']['counts']['validation']
    test_n = meta['split_metadata']['counts']['test']
    html.append(f'<header><button class="print" onclick="window.print()">In / Lưu PDF</button><div class="eyebrow">Lab 01 · Introduction to Machine Learning</div><h1>{escape(title)}</h1>'
                f'<p>Phân tích khám phá, kiểm tra chất lượng dữ liệu và bằng chứng cho thiết kế Linear Regression.</p>'
                f'<p class="meta">Kết quả chạy: {run_time:%d/%m/%Y %H:%M:%S} (UTC+7) · nguồn CSV: {escape(Path(meta["dataset_path"]).name)}</p></header>')
    markdown.extend([f'# {title}', '', f'Kết quả chạy: {run_time:%d/%m/%Y %H:%M:%S} (UTC+7).', ''])
    html.append('<div class="stats">'+''.join(f'<div class="stat"><strong>{value}</strong><span>{label}</span></div>'
        for value,label in [(f'{meta["row_count_raw"]:,}','dòng dữ liệu gốc'),(str(meta['columns_raw']),'thuộc tính'),
                            (f'{train_n:,}','dòng train để EDA'),(f'{val_n:,}','dòng validation')])+'</div>')
    source = meta['source_metadata']
    paragraph(f'Dữ liệu {source.get("provenance","unverified")}; giấy phép {source.get("license","unverified")}; '
              f'publisher version {source.get("dataset_version","unverified")}. '
              f'Test ({test_n:,} dòng) chỉ được dùng cho counts và integrity, không dùng target distribution, fit hoặc metrics. '
              'Các mô hình có biến AI/hành vi là conditional / retrospective do timing chưa xác minh. Không suy luận nhân quả.', True)

    titles = ['Tóm tắt và phạm vi', 'Chất lượng dữ liệu và split', 'Thống kê mô tả và phân phối', 'Target GPA và thay đổi GPA',
              'Quan hệ hai biến và nhóm sinh viên', 'Quan hệ đa biến và subgroup', 'Hồi quy thăm dò trên validation',
              'Đa cộng tuyến, phần dư và influence', 'Feature engineering', 'Leakage, provenance và bias',
              'Kết luận theo câu hỏi nghiên cứu', 'Khuyến nghị preprocessing và mô hình']
    html.append('<nav class="toc">'+''.join(f'<a href="#s{i}">{i:02}. {escape(name)}</a>' for i,name in enumerate(titles,1))+'</nav>')

    section(1,titles[0])
    results = t('T29').set_index('model')
    e1,e2,e3 = results.loc['E1'],results.loc['E2'],results.loc['E3']
    paragraph('Mục tiêu là dự đoán Post_Semester_GPA trước khi biết GPA cuối kỳ. Pre_Semester_GPA là baseline học tập; '
              'GPA_Change = Post − Pre chỉ là outcome phân tích, tuyệt đối không dùng làm predictor. '
              'Phân tích chi tiết dùng train; các so sánh mô hình định trước dùng cùng validation rows.')
    paragraph(f'Pre-only OLS: validation RMSE={e1.RMSE:.4f}, R²={e1.R2:.4f}. Academic OLS: RMSE={e2.RMSE:.4f}. '
              f'Academic + AI OLS: RMSE={e3.RMSE:.4f}, R²={e3.R2:.4f}. '
              f'E3 − E2 có ΔRMSE={e3.RMSE-e2.RMSE:+.5f}, ΔR²={e3.R2-e2.R2:+.5f}. '
              'Giá trị bổ sung được ghi nhận trong retrospective scenario; chưa chứng minh lợi ích forecast khi triển khai. [T29]')
    html.append('<p class="meta">Nguồn: <a href="'+escape(source['source_url'])+'">Kaggle — AI Impact on Students</a>. '
                'Các thống kê trong báo cáo được tính từ CSV cục bộ, không sao chép thống kê của publisher.</p>')
    table('T01',['rows','columns','file_bytes','missing_optional','extra_columns','encoding'])
    end_section()

    section(2,titles[1])
    missing = int(t('T05').missing_count.sum())
    excluded = meta['split_metadata']['counts'].get('excluded',0)
    paragraph(f'Structural audit toàn file: {missing:,} ô blank/missing; {excluded:,} dòng bị loại khỏi regression population. '
              'Không coi 0 là missing. Numeric parsing, whitespace/case mapping và domain rules được log. '
              'Chỉ exact duplicates bị loại; các ID/content liên quan được group trước split. '
              f'Split thực tế: train={train_n:,}, validation={val_n:,}, test={test_n:,}; seed={meta["config"]["seed"]}. [T05–T09, T41]')
    table('T05',['feature','missing_count','missing_pct','denominator'])
    table('T06')
    invalid = t('T07')
    affected = invalid[invalid.affected_rows.gt(0)]
    paragraph(f'T07 có {len(invalid)} domain/coercion checks; {len(affected)} checks ghi nhận vi phạm. '
              'Predictor invalid được chuyển missing để impute train-only; target invalid không được impute. '
              'Outlier hợp lệ được giữ lại, không tự động xóa.')
    table('T08',['feature','valid_n','iqr','lower_fence','upper_fence','flagged_count','flagged_pct','reason'])
    figure('F55','Counts từ frozen assignments; không chứa phân phối target test.')
    end_section()

    section(3,titles[2])
    paragraph('Thống kê chỉ dùng train. Sample std/variance dùng ddof=1; kurtosis là excess Fisher (normal=0). '
              'Ordinal được xem thêm bằng tần suất; mean của numeric rating không chứng minh khoảng cách thang đo bằng nhau. '
              'Predictors không cần phân phối chuẩn để fit OLS.')
    table('T10',['feature','valid_n','mean','median','std','minimum','maximum','q1','q3','skewness'])
    hours = t('T10').set_index('feature').loc['Weekly_GenAI_Hours']
    paragraph(f'Weekly GenAI Hours có mean={hours["mean"]:.3f}, median={hours["median"]:.3f} giờ/tuần, '
              f'skewness={hours.skewness:.3f}. Phân phối lệch phải là lý do khảo sát log/centering, chưa phải lý do loại biến. [T10/T35]')
    figure('F06','Histogram của các biến continuous, bao gồm target và auxiliary outcomes đúng vai trò.')
    figure('F10','Ordinal/discrete dùng bar chart có thứ tự; không áp dụng KDE cho category codes.')
    end_section()

    section(4,titles[3])
    change = t('T15').set_index('statistic').value
    paragraph(f'Paired valid n={int(change["count"]):,}; mean GPA Change={change["mean"]:.4f}, '
              f'median={change["50%"]:.4f}, sample std={change["std"]:.4f}. '
              'Improved/unchanged/declined dùng tolerance 1e-9 để xử lý sai số float; chưa phải ngưỡng thay đổi có ý nghĩa học thuật. [T15/T16]')
    table('T16')
    table('T18',['feature','boundary_type','boundary','count','percentage','valid_n','reason'])
    paragraph('Ceiling chỉ xét upper endpoint 4.0 được mô tả bởi nguồn; theoretical floor chưa được xác minh. '
              'GPA Change có mathematical coupling với Pre GPA và có thể chịu ceiling/regression to the mean; không đồng nghĩa tác động của AI.')
    figure('F12','Phân phối Post GPA trên train.')
    figure('F14','Thay đổi GPA quan sát trên cùng các paired training rows.')
    figure('F16','Pre và Post cùng thang trục, kèm identity line y=x; màu biểu thị count.')
    end_section()

    section(5,titles[4])
    paragraph('Pearson dùng quantitative features; Spearman dùng quantitative/ordinal có thứ tự được xác minh. '
              'Nominal không được gán số tùy ý. Bảng nêu valid pair count; liên hệ marginal không tự quyết định feature selection.')
    table('T19',['feature','outcome','valid_n','coefficient','reason'])
    table('T20',['feature','outcome','valid_n','coefficient','reason'])
    figure('F24','Linear fit và train-decile binned trend; scatter sampling có seed, correlations dùng toàn bộ valid train pairs.')
    groups = t('T21')
    table('T21',['feature','group','valid_n','mean','median','std','ci95_low','ci95_high','small_group'],
          groups[groups.feature.isin(['Prompt_Engineering_Skill','Primary_Use_Case']) & groups.outcome.eq('Post_Semester_GPA')],
          'Prompt Skill / Primary Use Case vs Post GPA')
    paragraph('95% t-based mean CIs là descriptive và phụ thuộc giả định independence chưa xác minh. '
              f'Nhóm n<{meta["config"]["min_group"]} được flag. Chênh lệch nhóm chưa điều chỉnh không được diễn giải là causal effect.')
    figure('F27','Phân phối GPA theo mục đích AI, group n trong T21.')
    figure('F28','Phân phối GPA theo mức Prompt Skill, không giả định equal ordinal spacing.')
    end_section()

    section(6,titles[5])
    paragraph('Phân tầng trên train để khảo sát AI Hours × Prompt Skill, Major, Study Hours, Pre GPA, Dependency; '
              'Use Case × Prompt; Dependency × Anxiety; và auxiliary Retention. '
              'Bins/edges được lưu ở T25; từng grouped trend có count và bounds ở T26. '
              f'Heatmap mask cell n<{meta["config"]["min_group"]}; không tô 0 vào ô thiếu.')
    table('T23',['group','valid_n','mean','median','std','ci95_low','ci95_high','small_group'])
    figure('F35','Mỗi ô: mean GPA và valid n. Hai chiều được sắp theo thứ tự quantile thực tế.')
    figure('F36','Use Case × Prompt Skill: liên hệ có điều kiện theo subgroup, chưa phải causal moderation.')
    figure('F38','GPA Change theo AI Hours và Pre GPA groups; vẫn có mathematical coupling với Pre GPA.')
    figure('F39','Retention chỉ là auxiliary outcome, tuyệt đối không nằm trong predictor matrix.')
    end_section()

    section(7,titles[6])
    paragraph('Fit train-only: numeric median; categorical mode + one-hot drop-first, xử lý unknown categories. '
              'E0a là train-mean; E0b persistence Pre GPA; E1 Pre-only; E2 academic; E3 academic + AI. '
              'Mọi model dùng cùng validation rows, không tuning cuối cùng. E2/E3 thêm các biến timing unknown nên có nhãn conditional / retrospective.')
    table('T29',['model','scenario','valid_n','MAE','MSE','RMSE','R2','adjusted_R2_train','delta_reference','delta_RMSE','delta_R2'])
    paragraph('Adjusted R² thuộc TRAIN fitted sample; các metrics MAE/MSE/RMSE/R² ở cột đánh giá thuộc VALIDATION. '
              'R² âm được giữ nếu xảy ra. Raw coefficients khác đơn vị không phải feature importance; reference categories có trong T30 và preprocessing_log.csv.')
    figure('F49','So sánh validation exploratory trên cùng sample population. Không có test metrics.')
    end_section()

    section(8,titles[7])
    vif = t('T28')
    rank = vif.iloc[0]
    paragraph(f'Encoded design: n={int(rank.n):,}, rank={int(rank["rank"])}/{int(rank.columns_including_intercept)}; '
              f'max VIF={vif.VIF.max():.4f}; condition number raw={rank.condition_number_raw:.3f}, '
              f'standardized={rank.condition_number_standardized:.3f}. '
              'VIF xét từng encoded term có intercept; ngưỡng 5/10 là heuristic, không phải quy tắc xóa. [T28]')
    table('T28',['encoded_term','original_feature','VIF','heuristic'],vif.sort_values('VIF',ascending=False).head(15),
          '15 encoded terms có VIF cao nhất; toàn bộ terms trong CSV')
    table('T31')
    paragraph('Residual = actual − fitted. Standardized residual là loại internally standardized; '
              '|r|>3, h>2p/n và Cook D>4/n chỉ để điều tra. Không tự động loại observations. '
              'Independence cần sampling/design evidence; residual normality liên quan inference, không phải predictor normality.')
    figure('F43','Residual vs fitted và binned mean sau khi fit conditional OLS.')
    figure('F45','Q-Q của internally standardized train residuals; xem departure thực tiễn cùng quy mô mẫu.')
    figure('F46','Scale-location để quan sát variance theo fitted GPA; không thay cho xác minh independence.')
    end_section()

    section(9,titles[8])
    paragraph('Centering fit bằng train mean; log1p chỉ cho H≥0. Polynomial giữ lower-order H; '
              'không ghép raw H và centered H với cùng intercept. Ordinal numeric products chưa có interval-scale evidence nên chỉ proposal. '
              'Total Study Hours và AI Study Ratio bị skip khi chưa xác minh AI usage chỉ dành cho study và không double-count.')
    table('T33')
    table('T35',['candidate','valid_n','skewness','pearson_post','spearman_post','pair_vif_with_linear_H',
                 'validation_MAE','validation_RMSE','validation_R2','quadratic_coefficient'])
    quad = t('T35').set_index('candidate').loc['Centered_AI_Hours_Squared']
    paragraph(f'Fixed E3 + centered H²: validation RMSE={quad.validation_RMSE:.5f}, '
              f'ΔRMSE so với E3={quad.validation_RMSE-e3.RMSE:+.6f}; quadratic coefficient={quad.quadratic_coefficient:.7f}. '
              'Đây là một comparison định trước, chưa chọn degree hay tuning theo validation. Giảm skewness chưa đồng nghĩa tăng khả năng dự đoán. [T35]')
    figure('F50','Phân phối original/log/centered-square; transformations không tự động được chọn vào final pipeline.')
    figure('F53','Quan hệ marginal train linear vs quadratic. Comparison validation ở T35 có academic/AI controls của E3.')
    end_section()

    section(10,titles[9])
    paragraph('Loại target, GPA Change, Student ID, auxiliary Skill Retention/Burnout và các cột không rõ vai trò khỏi X. '
              'Pre GPA có định nghĩa trước kỳ; timing của đa số biến khác chưa rõ. '
              'Synthetic provenance được xác nhận từ metadata; tính đại diện của cohort thật và generator mechanism vẫn unknown.')
    table('T40',['feature','availability','allowed_scenario','forecast_eligible','leakage_risk'])
    table('T41',['check','left','right','value','passed'])
    table('T42',['issue','status','consequence','action'])
    paragraph('Hai PDF tham chiếu trong brief chưa có trong workspace; cần xác minh nguyên văn rubric, số thành viên và Ridge allowance. '
              'Local CSV có fingerprint cụ thể, nhưng chưa có publisher checksum xác nhận phiên bản byte.')
    end_section()

    section(11,titles[10])
    for row in t('T44').fillna('').itertuples():
        html.append(f'<article class="finding"><h3>{escape(row.finding_id)} / {escape(row.research_question)}</h3>'
                    f'<p class="meta">Scope: {escape(row.data_scope)} · status: {escape(row.status)}</p>')
        markdown.extend([f'### {row.finding_id} / {row.research_question}', '', f'Scope: {row.data_scope}; status: {row.status}.', ''])
        for label,value in [('Observation',row.observation),('Interpretation',row.interpretation),
                            ('Recommended action',row.recommended_action),('Limitation',row.limitation)]:
            html.append(f'<p><strong>{label}:</strong> {escape(str(value))}</p>')
            markdown.extend([f'**{label}:** {value}', ''])
        refs = []
        for aid in row.evidence_ids.split():
            record = manifest.loc[aid]
            refs.append(f'<a href="{escape(record.relative_path)}">{aid}</a>' if record.relative_path else escape(aid))
        html.append('<p class="evidence">Evidence: '+', '.join(refs)+'</p></article>')
        markdown.extend(['Evidence: '+row.evidence_ids, ''])
    end_section()

    section(12,titles[11])
    table('T45',['feature','semantic_type','availability','train_missing_pct','encoding','transformation','decision','reason'])
    table('T47')
    paragraph('Training tiếp theo: cố định eligible feature set theo forecast hoặc retrospective scenario; '
              'fit imputer/encoder/scaler trong train/CV; giữ baseline mean/persistence; '
              'so sánh MAE, MSE, RMSE, R² trên cùng target/splits/population. '
              'Chỉ mở final test sau khi specification và lựa chọn mô hình đã đóng băng. '
              'Số model tối thiểu phải phù hợp số thành viên thực tế; Ridge cần được môn học chấp nhận.')
    html.append('<p>Xem thêm <a href="PREPROCESSING_PLAN.md">Preprocessing Plan đầy đủ</a>, '
                '<a href="DATA_DICTIONARY.md">Data Dictionary</a>, '
                '<a href="VERIFICATION.md">Verification</a> và '
                '<a href="01_EDA.executed.ipynb">executed notebook</a>.</p>')
    end_section()

    html.append('<section id="artifacts"><h2>Phụ lục — Artifact index và kiểm chứng</h2>')
    manifest_counts = manifest.groupby(['artifact_type','status']).size()
    paragraph('; '.join(f'{kind}/{status}: {count}' for (kind,status),count in manifest_counts.items())+'. '
              'Skipped và alias được giữ nguyên, không tính file thiếu là generated. '
              f'Fresh-kernel execution: {verification["clean_kernel_execution"]}; '
              f'reproducible rerun: {verification["reproducible_rerun"]}.')
    verification_frame = pd.DataFrame([{'criterion':key, **value} for key,value in verification['acceptance_criteria'].items()])
    html.append('<div class="table-wrap">'+verification_frame.to_html(index=False,border=0,escape=True)+'</div>')
    markdown.extend(['## Phụ lục — Kiểm chứng', '', verification_frame.to_markdown(index=False), ''])
    links = manifest.reset_index()[['artifact_id','title','section','data_scope','status','relative_path','related_ids','reason']]
    link_html = links.copy()
    link_html['relative_path'] = link_html.relative_path.map(lambda p:f'<a href="{escape(p)}">Mở file</a>' if p else '—')
    for c in link_html.columns.difference(['relative_path']):
        link_html[c] = link_html[c].map(lambda value:escape(str(value)))
    html.append('<details><summary>Toàn bộ 105 artifact IDs</summary><div class="table-wrap">'+link_html.to_html(index=False,border=0,escape=False)+'</div></details></section>')
    markdown.extend(['## Artifact index', '', links.to_markdown(index=False), ''])
    footer = f'Dataset SHA-256: {meta["dataset_sha256"]} · Config SHA-256: {meta["config_sha256"]}. '
    html.append('<footer>'+escape(footer)+'<br>Báo cáo được dựng từ artifacts của notebook vừa chạy; các hình trong bản HTML được nhúng để đọc offline. '
                'Links CSV/notebook chi tiết cần đi kèm thư mục reports.</footer>')
    markdown.extend(['', footer, ''])
    document = '<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    document += '<title>'+escape(title)+'</title><style>'+CSS+'</style></head><body><main>'+''.join(html)+'</main></body></html>'
    (output_dir/'EDA_REPORT.html').write_text(document,encoding='utf-8')
    (output_dir/'EDA_REPORT.md').write_text('\n'.join(markdown),encoding='utf-8')
    provenance = {'dataset_sha256':meta['dataset_sha256'],'config_sha256':meta['config_sha256'],
                  'source_run_completed_at':meta['completed_at'],'source_artifacts':sorted(used),
                  'html_sha256':hashlib.sha256((output_dir/'EDA_REPORT.html').read_bytes()).hexdigest(),
                  'artifacts_used_sha256':{aid:hashlib.sha256((output_dir/manifest.loc[aid,'relative_path']).read_bytes()).hexdigest() for aid in sorted(used)}}
    (output_dir/'EDA_REPORT.metadata.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps({'status':'generated','report_html':str(output_dir/'EDA_REPORT.html'),
                      'report_markdown':str(output_dir/'EDA_REPORT.md'),'artifacts_used':len(used),
                      'html_bytes':(output_dir/'EDA_REPORT.html').stat().st_size},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reports')
    render(parser.parse_args().output_dir)
