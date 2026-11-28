import { useEffect, useState } from 'react'
import { ArrowDown, ArrowUp, ArrowUpRight, ChartNoAxesCombined, CircleHelp, LoaderCircle, RefreshCcw, TriangleAlert } from 'lucide-react'
import { request } from './api'
import { modelColors } from './model-catalog'
import type { Agreement, Bootstrap, Comparison, Profile, ValidationSummary } from './types'

function PredictionBars({ result }: { result: Comparison }) {
  const values = result.rows.flatMap(row => row.gpa === null ? [] : [row.gpa])
  const lo = Math.min(0, ...values), hi = Math.max(4, ...values)
  const x = (n: number) => 130 + (n - lo) / (hi - lo) * 390
  return <svg className="comparison-bars" viewBox="0 0 590 290" role="img" aria-label="GPA của sáu mô hình trên cùng hồ sơ">
    {[0, 1, 2, 3, 4].map(i => {
      const value = lo + (hi - lo) * i / 4
      return <g key={i}><line x1={x(value)} x2={x(value)} y1="8" y2="244" className="chart-grid" /><text x={x(value)} y="271" textAnchor="middle">{value.toFixed(1)}</text></g>
    })}
    {result.rows.map((row, i) => <g key={row.model}>
      <text x="116" y={30 + i * 38} textAnchor="end">{row.model}{row.model === 'Simple' ? ` · ${result.variant}` : ''}</text>
      {row.gpa !== null ? <><rect x={Math.min(x(0), x(row.gpa))} y={17 + i * 38}
        width={Math.max(1, Math.abs(x(row.gpa) - x(0)))} height="19" rx="4" fill={modelColors[i]} />
        <text x={x(row.gpa) + 8} y={30 + i * 38}>{row.gpa.toFixed(3)}</text></> :
        <text x="140" y={30 + i * 38}>Chưa sẵn sàng</text>}
    </g>)}
    <text x="325" y="289" textAnchor="middle">GPA dự đoán · giữ nguyên giá trị raw</text>
  </svg>
}

function AgreementChart({ data }: { data: Agreement }) {
  const [lo, hi] = data.domain
  const x = (n: number) => 50 + (n - lo) / (hi - lo || 1) * 380
  const y = (n: number) => 290 - (n - lo) / (hi - lo || 1) * 260
  return <svg className="agreement-chart" viewBox="0 0 480 340" role="img" aria-label={`Thực tế so với dự đoán trên validation · ${data.model}`}>
    {[0, 1, 2, 3, 4].map(i => {
      const value = lo + (hi - lo) * i / 4
      return <g key={i}><line x1="50" x2="430" y1={y(value)} y2={y(value)} className="chart-grid" />
        <line x1={x(value)} x2={x(value)} y1="30" y2="290" className="chart-grid" />
        <text x={x(value)} y="310" textAnchor="middle">{value.toFixed(1)}</text>
        <text x="40" y={y(value) + 4} textAnchor="end">{value.toFixed(1)}</text></g>
    })}
    <line x1={x(lo)} x2={x(hi)} y1={y(lo)} y2={y(hi)} stroke="var(--amber)" strokeDasharray="5 5" strokeWidth="1.5" />
    {data.points.map((point, i) => <circle key={i} cx={x(point.actual)} cy={y(point.predicted)} r="2.6" fill="var(--teal)" opacity=".4">
      <title>Thực tế {point.actual.toFixed(3)} · Dự đoán {point.predicted.toFixed(3)}</title>
    </circle>)}
    <text x="50" y="16">GPA dự đoán</text><text x="240" y="335" textAnchor="middle">GPA thực tế</text>
  </svg>
}

export default function ComparisonPage({ data, profile, variant, onVariantChange, result, dirty, pending, busy, error, onRun, onEdit }: {
  data: Bootstrap; profile: Profile; variant: string; onVariantChange: (variant: string) => void
  result: Comparison | null; dirty: boolean; pending: boolean; busy: boolean
  error: string; onRun: () => void; onEdit: () => void
}) {
  const [validation, setValidation] = useState<ValidationSummary | null>(null)
  const [validationError, setValidationError] = useState('')
  const [validationRetry, setValidationRetry] = useState(0)
  const [scatterModel, setScatterModel] = useState('Interaction')
  const [agreement, setAgreement] = useState<Agreement | null>(null)
  const [agreementError, setAgreementError] = useState('')
  const [agreementRetry, setAgreementRetry] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    setValidationError('')
    request<ValidationSummary>('/api/validation', { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) setValidation(value) })
      .catch(err => { if (!controller.signal.aborted) setValidationError(err.message) })
    return () => controller.abort()
  }, [validationRetry])
  const eligible = validation?.metrics.filter(row => row.eligible).map(row => row.model) ?? []
  const selectedScatter = eligible.includes(scatterModel) ? scatterModel : eligible[0]
  useEffect(() => {
    if (!selectedScatter) return
    const controller = new AbortController()
    setAgreement(null); setAgreementError('')
    request<Agreement>(`/api/agreement?model=${encodeURIComponent(selectedScatter)}`, { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) setAgreement(value) })
      .catch(err => { if (!controller.signal.aborted) setAgreementError(err.message) })
    return () => controller.abort()
  }, [selectedScatter, agreementRetry])
  return <div className="comparison-page">
    <section className="page-heading comparison-heading"><div><p className="eyebrow">PHÂN TÍCH / 02</p><h1>Đặt các mô hình <span>cạnh nhau.</span></h1>
      <p>So sánh kết quả của cùng một hồ sơ, sau đó kiểm tra sai số trên tập validation.</p></div></section>
    <div className="comparison-content">
      <section className="comparison-section">
        <div className="comparison-section-heading"><div><p className="eyebrow">THÍ NGHIỆM / 01</p><h2>Cùng một hồ sơ, sáu góc nhìn</h2></div>
          <a href="#prediction" onClick={e => { e.preventDefault(); onEdit() }} className="edit-profile">Chỉnh sửa hồ sơ<ArrowUpRight size={15} /></a>
        </div>
        <div className="comparison-workspace">
          <div className="comparison-config">
            <div className="profile-summary"><p className="eyebrow">HỒ SƠ ĐANG DÙNG</p>
              <dl><div><dt>GPA trước kỳ</dt><dd>{String(profile.Pre_Semester_GPA)}</dd></div>
                <div><dt>Giờ tự học</dt><dd>{String(profile.Traditional_Study_Hours)} h</dd></div>
                <div><dt>Giờ AI / tuần</dt><dd>{String(profile.Weekly_GenAI_Hours)} h</dd></div></dl>
              <p>{String(profile.Year_of_Study)} · {String(profile.Major_Category)}</p>
            </div>
            <div className="compare-config-panel">
              <label htmlFor="comparison-variant">Biến thể Simple</label>
              <select id="comparison-variant" value={variant} disabled={busy} onChange={e => onVariantChange(e.target.value)}>
                {Object.entries(data.simple_variants).map(([id, field]) =>
                  <option key={id} value={id}>{id} · {data.fields.find(f => f.id === field)?.label}</option>)}
              </select>
              <button className="predict-button" disabled={busy} onClick={onRun}>
                {pending ? <LoaderCircle className="spin" size={17} /> : <RefreshCcw size={17} />}
                {pending ? 'Đang so sánh…' : 'So sánh 6 mô hình'}<ArrowUpRight size={16} />
              </button>
              <p>Chỉ chạy khi bạn nhấn nút. Thay đổi hồ sơ hoặc biến thể không tự tính lại.</p>
            </div>
            <div className="comparison-policy"><CircleHelp size={17} /><p><strong>Giữ nguyên giá trị raw.</strong> GPA ngoài miền 0–4 không bị cắt. GPA cao hơn không có nghĩa mô hình chính xác hơn.</p></div>
          </div>
          <div className="comparison-output" aria-busy={pending}>
            {dirty && <p className="dirty-notice" role="status"><TriangleAlert size={17} /><span><strong>Kết quả lần so sánh trước · Simple {result?.variant}.</strong><br />Hồ sơ hoặc biến thể đã thay đổi. Nhấn so sánh để cập nhật.</span></p>}
            {error && <p className="error-notice" role="alert"><TriangleAlert size={17} />{error}</p>}
            {pending && <p className="comparison-loading" role="status"><LoaderCircle className="spin" size={19} />Đang chạy sáu mô hình trên cùng hồ sơ…</p>}
            {result ? <div className="comparison-result">
              <div className="section-caption"><span>GPA CỦA CÙNG HỒ SƠ</span><span>KẾT QUẢ PYTHON</span></div>
              <p className="snapshot-caption">Hồ sơ đã so sánh: GPA {String(result.profile.Pre_Semester_GPA)} · AI {String(result.profile.Weekly_GenAI_Hours)}h · Simple {result.variant}</p>
              <PredictionBars result={result} />
              <div className="table-scroll" tabIndex={0} role="region" aria-label="Bảng GPA có thể cuộn ngang">
                <table aria-label="GPA của cùng hồ sơ"><thead><tr><th>Mô hình</th><th>GPA dự đoán</th><th>Trạng thái</th></tr></thead><tbody>
                  {result.rows.map(row => <tr key={row.model}><th scope="row">{row.model}{row.model === 'Simple' ? ` · ${result.variant}` : ''}</th>
                    <td>{row.gpa === null ? '—' : row.gpa.toFixed(3)}</td><td>{row.error ?? (row.out_of_range ? 'Ngoài 0–4 · không cắt' : 'Đã dự đoán')}</td></tr>)}
                </tbody></table>
              </div>
            </div> : !pending && <div className="comparison-empty"><ChartNoAxesCombined size={38} /><h3>Chưa có kết quả so sánh</h3><p>Nhấn “So sánh 6 mô hình” để đối chiếu trên cùng hồ sơ.</p></div>}
          </div>
        </div>
      </section>
      <section className="comparison-section">
        <div className="comparison-section-heading"><div><p className="eyebrow">ĐÁNH GIÁ / {validation?.n_rows.toLocaleString('vi-VN') ?? '…'} DÒNG</p><h2>Hiệu năng trên tập validation</h2></div><span className="validation-tag">TEST KHÔNG THAM GIA</span></div>
        <p className="comparison-description">Đánh giá chất lượng mô hình trên dữ liệu validation — không phải điểm số của hồ sơ đang nhập.</p>
        {validationError ? <div className="error-notice" role="alert">{validationError}<button onClick={() => setValidationRetry(validationRetry + 1)}>Thử lại</button></div> :
          !validation ? <p role="status" className="comparison-loading"><LoaderCircle className="spin" size={17} />Đang tính chỉ số validation…</p> :
          <div className="validation-layout"><div className="table-scroll validation-table" tabIndex={0} role="region" aria-label="Bảng chỉ số có thể cuộn ngang">
            <table aria-label="Chỉ số validation"><thead><tr><th>Mô hình</th><th>MAE ↓</th><th>RMSE ↓</th><th>R² ↑</th><th>Trạng thái</th></tr></thead><tbody>
              {validation.metrics.map(row => <tr key={row.model} className={row.eligible ? '' : 'excluded-metric'}>
                <th scope="row">{row.model}</th><td>{row.mae?.toFixed(6) ?? '—'}</td><td>{row.rmse?.toFixed(6) ?? '—'}</td><td>{row.r2?.toFixed(6) ?? '—'}</td>
                <td>{row.eligible ? <span className="metric-ready">Đã xác minh</span> : <><strong>Không đánh giá</strong><small>{row.reason}</small></>}</td>
              </tr>)}
            </tbody></table>
          </div><aside className="metric-guide"><h3>Đọc các chỉ số</h3><p><ArrowDown size={16} /><strong>MAE & RMSE</strong><span>Càng thấp càng tốt. Đơn vị sai số là điểm GPA.</span></p>
            <p><ArrowUp size={16} /><strong>R²</strong><span>Càng cao càng tốt; có thể âm. Không phải phần trăm chính xác.</span></p>
            <p className="guide-note">Simple đã fit trên train + validation, nên không chấm lại trên validation.</p>
          </aside></div>}
      </section>
      <section className="comparison-section scatter-section">
        <div className="comparison-section-heading"><div><p className="eyebrow">ĐỘ LỆCH / 03</p><h2>Thực tế so với dự đoán</h2></div>
          <div className="scatter-selector"><label htmlFor="scatter-model">Mô hình trên biểu đồ</label>
            <select id="scatter-model" disabled={!eligible.length} value={selectedScatter ?? ''} onChange={e => setScatterModel(e.target.value)}>
              {eligible.map(model => <option key={model} value={model}>{model}</option>)}
            </select></div>
        </div>
        {agreementError ? <div className="error-notice" role="alert">{agreementError}<button onClick={() => setAgreementRetry(agreementRetry + 1)}>Thử lại</button></div> :
          agreement ? <div className="scatter-content"><AgreementChart data={agreement} /><div className="scatter-explanation"><span className="metric-ready">DỮ LIỆU VALIDATION THẬT</span>
            <h3>{agreement.model}</h3><p>Hiển thị {agreement.points.length} điểm lấy mẫu cố định. Các chỉ số dùng đủ {agreement.n_rows.toLocaleString('vi-VN')} dòng validation.</p>
            <div className="chart-legend"><span><i className="dot" />Dự đoán của mô hình</span><span><i />Đường dự đoán hoàn hảo</span></div>
            <p>Trục biểu đồ bao gồm cả dự đoán ngoài 0–4, không cắt giá trị.</p></div></div> :
            <p className="comparison-loading" role="status">{selectedScatter ? 'Đang tải biểu đồ validation…' : 'Biểu đồ cần mô hình đủ điều kiện đánh giá.'}</p>}
      </section>
      <footer className="scientific-note"><CircleHelp size={17} /><p>Dữ liệu tổng hợp. Kết quả không phải GPA thực tế của sinh viên và không chứng minh AI gây thay đổi điểm số.</p></footer>
    </div>
  </div>
}
