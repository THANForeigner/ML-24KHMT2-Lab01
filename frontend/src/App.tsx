import { useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import {
  ArrowUpRight, ChartNoAxesCombined, Check, ChevronDown,
  ChevronLeft, ChevronRight, CircleHelp, FlaskConical, Layers, LayoutGrid,
  LineChart, LoaderCircle, Moon, PanelLeftClose, PanelLeftOpen, RefreshCcw,
  SlidersHorizontal, Sparkles, Sun, TriangleAlert,
} from 'lucide-react'
import type { Bootstrap, Comparison, Field, Model, Page, Prediction, PredictionInput, Profile } from './types'
import { request } from './api'
import { modelIcons as icons } from './model-catalog'
import OverviewPage from './OverviewPage'
import ComparisonPage from './ComparisonPage'

const groups = ['Thông tin học tập', 'Thói quen sử dụng AI', 'Bối cảnh cá nhân']
const groupNotes = ['Nền tảng trước kỳ dự đoán.', 'Cách bạn học cùng công nghệ.', 'Những yếu tố trong hồ sơ sinh viên.']
const curveCandidates = ['Weekly_GenAI_Hours', 'Pre_Semester_GPA', 'Traditional_Study_Hours', 'Tool_Diversity', 'Perceived_AI_Dependency']
const vietnamese: Record<string, string> = {
  Freshman: 'Năm nhất', Sophomore: 'Năm hai', Junior: 'Năm ba', Senior: 'Năm tư', Graduate: 'Sau đại học',
  Arts: 'Nghệ thuật', Business: 'Kinh doanh', Humanities: 'Khoa học nhân văn', Medical: 'Y khoa', STEM: 'Khoa học, công nghệ, kỹ thuật & toán',
  'Copywriting/Drafting': 'Soạn thảo nội dung', 'Debugging/Troubleshooting': 'Gỡ lỗi và khắc phục sự cố',
  Direct_Answer_Generation: 'Tạo câu trả lời trực tiếp', Ideation: 'Lên ý tưởng', Summarizing_Reading: 'Tóm tắt tài liệu',
  Beginner: 'Mới bắt đầu', Intermediate: 'Trung cấp', Advanced: 'Nâng cao',
  Strict_Ban: 'Cấm sử dụng', Allowed_With_Citation: 'Cho phép nếu trích dẫn', Actively_Encouraged: 'Khuyến khích sử dụng',
}
function readPreference(key: string) {
  try { return localStorage.getItem(key) } catch { return null }
}
function savePreference(key: string, value: string) {
  try { localStorage.setItem(key, value) } catch { /* Storage is optional. */ }
}
function routePage(): Page {
  const hash = window.location.hash.slice(1)
  return hash === 'prediction' || hash === 'comparison' ? hash : 'overview'
}

function ModelPicker({ models, index, onChange, disabled }: {
  models: Model[]; index: number; onChange: (index: number) => void; disabled: boolean
}) {
  const [open, setOpen] = useState(false)
  const [focused, setFocused] = useState(index)
  const wrapper = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const options = useRef<(HTMLButtonElement | null)[]>([])
  const Icon = icons[index]
  useEffect(() => {
    const outside = (event: PointerEvent) => {
      if (!wrapper.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', outside)
    return () => document.removeEventListener('pointerdown', outside)
  }, [])
  useEffect(() => { if (open) options.current[focused]?.focus() }, [open, focused])
  function close() { setOpen(false); trigger.current?.focus() }
  function select(value: number) { onChange(value); close() }
  function keys(event: KeyboardEvent, position: number) {
    if (event.key === 'Escape') { event.preventDefault(); close() }
    if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
      event.preventDefault()
      setOpen(true)
      setFocused(event.key === 'Home' ? 0 : event.key === 'End' ? models.length - 1 :
        (position + (event.key === 'ArrowDown' ? 1 : -1) + models.length) % models.length)
    }
    if (event.key === 'Tab') setOpen(false)
  }
  return <div className="model-switcher" ref={wrapper}>
    <div className="stepper">
      <button className="icon-button" aria-label="Mô hình trước" title="Mô hình trước"
        disabled={disabled || index === 0} onClick={() => onChange(index - 1)}><ChevronLeft size={18} /></button>
      <span className="model-position" aria-label={`Mô hình ${index + 1} trên ${models.length}`}>{index + 1} / {models.length}</span>
      <button className="icon-button" aria-label="Mô hình tiếp theo" title="Mô hình tiếp theo"
        disabled={disabled || index === models.length - 1} onClick={() => onChange(index + 1)}><ChevronRight size={18} /></button>
    </div>
    <button className="model-trigger" aria-label={`Chọn mô hình: ${models[index].id}`} ref={trigger}
      aria-haspopup="listbox" aria-controls="model-list" aria-expanded={open} disabled={disabled}
      onKeyDown={event => keys(event, index)}
      onClick={() => { setFocused(index); setOpen(!open) }}>
      <span className={`model-icon model-color-${index}`}><Icon size={19} /></span>
      <span className="model-trigger-text"><small>MÔ HÌNH ĐANG CHỌN</small><strong>{models[index].id}</strong></span>
      <ChevronDown size={16} className={open ? 'rotate' : ''} />
    </button>
    {open && <div className="model-popover" id="model-list" role="listbox" aria-label="Mô hình dự đoán">
      <p className="popover-heading">CHỌN CÁCH NHÌN DỮ LIỆU <span>06 MODELS</span></p>
      {models.map((model, i) => {
        const ModelIcon = icons[i]
        return <button key={model.id} type="button" role="option" aria-selected={i === index}
          ref={element => { options.current[i] = element }} tabIndex={i === focused ? 0 : -1}
          className={`model-option ${i === index ? 'selected' : ''}`}
          onKeyDown={event => keys(event, i)} onClick={() => select(i)}>
          <span className={`model-icon model-color-${i}`}><ModelIcon size={19} /></span>
          <span><strong>{model.id}</strong><small>{model.description}{!model.available ? ' · Chưa có artifact' : ''}</small></span>
          {i === index && <Check size={16} />}
        </button>
      })}
    </div>}
  </div>
}

function InputField({ field, value, unused, disabled, onChange }: {
  field: Field; value: Profile[string]; unused: boolean; disabled: boolean
  onChange: (value: Profile[string]) => void
}) {
  const options = field.options ?? []
  return <div className={`input-group ${unused ? 'unused' : ''}`}>
    <label htmlFor={field.id}>{field.label}</label>
    {field.kind === 'number' ?
      <input id={field.id} name={field.id} type="number" required value={String(value)}
        min={field.min} max={field.max} step={field.step} disabled={disabled}
        onChange={e => onChange(e.target.value === '' ? '' : Number(e.target.value))} /> :
      <div className="select-wrap"><select id={field.id} name={field.id} required disabled={disabled}
        value={String(value)} onChange={e => {
          const selected = options.find(option => String(option) === e.target.value)
          if (selected !== undefined) onChange(selected)
        }}>
        {options.map(option => <option key={String(option)} value={String(option)}>
          {typeof option === 'boolean' ? (option ? 'Có đăng ký' : 'Không đăng ký') :
            vietnamese[String(option)] ?? (field.id.includes('Dependency') || field.id.includes('Anxiety') ?
              `Mức ${option}/10` : field.id === 'Tool_Diversity' ? `${option} công cụ` : String(option))}
        </option>)}
      </select><ChevronDown size={14} aria-hidden /></div>}
    <small className="field-helper">{unused ? 'Không dùng trong mô hình này' : '\u00a0'}</small>
  </div>
}

function SensitivityChart({ result, label }: { result: Prediction; label: string }) {
  const [left, top, width, height] = [46, 20, 404, 204]
  const xs = result.curve.map(p => p.x)
  const ys = result.curve.map(p => p.gpa)
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const minY = Math.min(0, ...ys), maxY = Math.max(4, ...ys)
  const x = (n: number) => left + (n - minX) / (maxX - minX || 1) * width
  const y = (n: number) => top + height - (n - minY) / (maxY - minY || 1) * height
  const line = result.curve.map((p, i) => `${i ? 'L' : 'M'}${x(p.x)},${y(p.gpa)}`).join(' ')
  const currentX = x(Number(result.profile[result.variable]))
  return <svg className="sensitivity-chart" viewBox="0 0 480 280" role="img"
    aria-label={`Đường dự đoán GPA theo ${label}, mô hình ${result.model}`}>
    <defs><linearGradient id="chart-fill" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stopColor="var(--teal)" stopOpacity=".18" />
      <stop offset="100%" stopColor="var(--teal)" stopOpacity="0" />
    </linearGradient></defs>
    {[0, 1, 2, 3, 4].map(i => {
      const tick = minY + (maxY - minY) * i / 4
      return <g key={i}><line x1={left} x2={left + width} y1={y(tick)} y2={y(tick)} className="chart-grid" />
        <text x={left - 10} y={y(tick) + 4} textAnchor="end">{tick.toFixed(1)}</text>
      </g>
    })}
    <path d={`${line} L${left + width},${top + height} L${left},${top + height} Z`} fill="url(#chart-fill)" />
    <path d={line} fill="none" stroke="var(--teal)" strokeWidth="3" strokeLinejoin="round" />
    <line x1={currentX} x2={currentX} y1={top} y2={top + height} stroke="var(--amber)" strokeDasharray="4 4" />
    <circle cx={currentX} cy={y(result.gpa)} r="6" fill="var(--amber)" stroke="var(--surface)" strokeWidth="3">
      <title>Hồ sơ đã dự đoán: GPA {result.gpa.toFixed(3)}</title>
    </circle>
    {[0, 1, 2, 3, 4].map(i => {
      const tick = minX + (maxX - minX) * i / 4
      return <text key={i} x={x(tick)} y={top + height + 20} textAnchor="middle">{tick.toFixed(1)}</text>
    })}
    <text x={left} y="10">GPA dự đoán</text>
    <text x={left + width / 2} y="272" textAnchor="middle">{label}</text>
  </svg>
}

export default function App() {
  const [page, setPage] = useState<Page>(routePage)
  const [data, setData] = useState<Bootstrap | null>(null)
  const [profile, setProfile] = useState<Profile>({})
  const [modelIndex, setModelIndex] = useState(1)
  const [variant, setVariant] = useState('M03')
  const [variable, setVariable] = useState('Weekly_GenAI_Hours')
  const [result, setResult] = useState<Prediction | null>(null)
  const [submitted, setSubmitted] = useState('')
  const [pending, setPending] = useState(false)
  const [sampling, setSampling] = useState(false)
  const [error, setError] = useState('')
  const [sampleIndex, setSampleIndex] = useState(0)
  const [retry, setRetry] = useState(0)
  const [comparison, setComparison] = useState<Comparison | null>(null)
  const [comparisonSubmitted, setComparisonSubmitted] = useState('')
  const [comparing, setComparing] = useState(false)
  const [comparisonError, setComparisonError] = useState('')
  const [theme, setTheme] = useState(() => readPreference('lab01-theme') ??
    (typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'))
  const [collapsed, setCollapsed] = useState(() => readPreference('lab01-sidebar') === 'collapsed' ||
    (!readPreference('lab01-sidebar') && typeof matchMedia === 'function' && matchMedia('(max-width: 767px)').matches))
  const sidebarToggle = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' })
    document.getElementById('main-content')?.focus({ preventScroll: true })
  }, [page])
  useEffect(() => {
    const change = () => {
      if (['#overview', '#prediction', '#comparison', ''].includes(window.location.hash)) setPage(routePage())
    }
    window.addEventListener('hashchange', change)
    return () => window.removeEventListener('hashchange', change)
  }, [])
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    savePreference('lab01-theme', theme)
  }, [theme])
  useEffect(() => { savePreference('lab01-sidebar', collapsed ? 'collapsed' : 'expanded') }, [collapsed])
  useEffect(() => {
    const controller = new AbortController()
    setError('')
    request<Bootstrap>('/api/bootstrap', { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) { setData(value); setProfile(value.profile) } })
      .catch(err => { if (!controller.signal.aborted) setError(err.message) })
    return () => controller.abort()
  }, [retry])
  useEffect(() => {
    const escape = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape' && matchMedia('(max-width: 767px)').matches && !collapsed) {
        setCollapsed(true); sidebarToggle.current?.focus()
      }
    }
    document.addEventListener('keydown', escape)
    return () => document.removeEventListener('keydown', escape)
  }, [collapsed])
  const model = data?.models[modelIndex]
  const usedFields = model?.id === 'Simple' ? [data!.simple_variants[variant]] : model?.fields ?? []
  const variables = curveCandidates.filter(id => usedFields.includes(id))
  const selectedVariable = variables.includes(variable) ? variable : variables[0]
  const input: PredictionInput = { model: model?.id ?? '', variant, profile, variable: selectedVariable }
  const signature = JSON.stringify(input)
  const dirty = !!result && signature !== submitted
  const busy = pending || sampling || comparing
  const comparisonSignature = JSON.stringify({ profile, variant })
  const comparisonDirty = !!comparison && comparisonSignature !== comparisonSubmitted
  function navigate(next: Page) {
    setPage(next)
    if (window.location.hash !== `#${next}`) window.location.hash = next
    if (typeof matchMedia === 'function' && matchMedia('(max-width: 767px)').matches) setCollapsed(true)
  }
  function changeModel(index: number) { setModelIndex(index); setError('') }
  async function compare() {
    if (busy) return
    const snapshot = structuredClone({ profile, variant })
    setComparing(true); setComparisonError('')
    try {
      const response = await request<Comparison>('/api/compare', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(snapshot),
      })
      setComparison(response); setComparisonSubmitted(JSON.stringify(snapshot))
    } catch (err) { setComparisonError(err instanceof Error ? err.message : 'Không thể so sánh. Hãy thử lại.') }
    finally { setComparing(false) }
  }
  async function submit(event: FormEvent) {
    event.preventDefault()
    if (busy || !model) return
    const snapshot = structuredClone(input)
    const snapshotSignature = JSON.stringify(snapshot)
    setPending(true); setError('')
    try {
      const response = await request<Prediction>('/api/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(snapshot),
      })
      setResult(response); setSubmitted(snapshotSignature)
    } catch (err) { setError(err instanceof Error ? err.message : 'Không thể dự đoán. Hãy thử lại.') }
    finally { setPending(false) }
  }
  async function loadSample() {
    setSampling(true); setError('')
    try {
      const response = await request<{ profile: Profile }>(`/api/sample?index=${sampleIndex + 1}`)
      setProfile(response.profile); setSampleIndex(sampleIndex + 1)
    } catch (err) { setError(err instanceof Error ? err.message : 'Không tải được hồ sơ mẫu.') }
    finally { setSampling(false) }
  }
  const cta = `Dự đoán với ${model?.id ?? ''}${model?.id === 'Simple' ? ` · ${variant}` : ''}`
  const resultVariableLabel = data?.fields.find(field => field.id === result?.variable)?.label ?? ''
  return <div className={`app-shell ${collapsed ? 'sidebar-collapsed' : ''}`}>
    <a href="#main-content" className="skip-link" onClick={e => { e.preventDefault(); document.getElementById('main-content')?.focus() }}>Chuyển đến nội dung chính</a>
    {!collapsed && <button className="sidebar-backdrop" aria-label="Đóng thanh điều hướng" onClick={() => setCollapsed(true)} />}
    <aside className="sidebar" id="sidebar">
      <div className="brand"><span className="brand-mark">L</span><div className="sidebar-label"><strong>LAB / 01</strong><small>MODEL EXPLORER</small></div></div>
      <button className="sidebar-toggle" ref={sidebarToggle} aria-controls="sidebar" aria-expanded={!collapsed}
        aria-label={collapsed ? 'Mở rộng thanh điều hướng' : 'Thu gọn thanh điều hướng'}
        onClick={() => setCollapsed(!collapsed)}>
        {collapsed ? <PanelLeftOpen size={19} /> : <PanelLeftClose size={19} />}
        <span className="sidebar-label">Thu gọn</span>
      </button>
      <nav aria-label="Điều hướng chính">
        {([
          { id: 'overview', label: 'Tổng quan', Icon: LayoutGrid },
          { id: 'prediction', label: 'Thử dự đoán', Icon: FlaskConical },
          { id: 'comparison', label: 'So sánh', Icon: Layers },
        ] as const).map(({ id, label, Icon }) =>
          <a key={id} className={`nav-item ${page === id ? 'active' : ''}`} href={`#${id}`}
            aria-current={page === id ? 'page' : undefined} aria-label={label}
            onClick={e => { e.preventDefault(); navigate(id) }}>
            <Icon size={19} /><span className="sidebar-label">{label}</span><span className="nav-tooltip">{label}</span>
          </a>)}
      </nav>
    </aside>
    <main id="main-content" tabIndex={-1}>
      <header className="toolbar">
        {data && page === 'prediction' ? <ModelPicker models={data.models} index={modelIndex} onChange={changeModel} disabled={busy} /> :
          <div className="page-breadcrumb">{page === 'comparison' ? <Layers size={18} /> : <LayoutGrid size={18} />}
            <span>{page === 'comparison' ? 'Bảng đối chiếu mô hình' : 'Tổng quan phòng thí nghiệm'}</span></div>}
        <div className="toolbar-actions">
          {page !== 'overview' && <button className="sample-button" disabled={!data || busy} onClick={loadSample}>
            <RefreshCcw size={15} className={sampling ? 'spin' : ''} /><span>Dữ liệu mẫu</span>
          </button>}
          <button className="theme-button" onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
            aria-label={`Chuyển sang chế độ ${theme === 'light' ? 'tối' : 'sáng'}`} aria-pressed={theme === 'dark'}>
            {theme === 'light' ? <Moon size={17} /> : <Sun size={17} />}<span>Chế độ</span>
          </button>
        </div>
      </header>
      {page === 'prediction' && <section className="page-heading">
        <div><p className="eyebrow"><span /> PHÒNG THÍ NGHIỆM / 01</p>
          <h1>Một hồ sơ. <span>Nhiều khả năng.</span></h1>
          <p>Điều chỉnh hồ sơ sinh viên. Chọn mô hình. Khám phá dự đoán GPA.</p>
        </div>
        <span className="edition-tag"><FlaskConical size={16} /> MODEL EXPLORER <span>V.01</span></span>
      </section>}
      {!data ? <section className="loading-screen" aria-live="polite">
        {error ? <><p role="alert">{error}</p><button onClick={() => setRetry(retry + 1)}>Thử kết nối lại</button></> :
          <><LoaderCircle className="spin" />Đang tải phòng thí nghiệm…</>}
      </section> : page === 'overview' ?
        <OverviewPage data={data} busy={busy} onNavigate={navigate} onExplore={index => { changeModel(index); navigate('prediction') }} /> :
        page === 'comparison' ?
        <ComparisonPage data={data} profile={profile} variant={variant} onVariantChange={setVariant}
          result={comparison} dirty={comparisonDirty} pending={comparing} busy={busy}
          error={comparisonError || error} onRun={compare} onEdit={() => navigate('prediction')} /> :
        <div className="workspace">
        <form id="profile-form" onSubmit={submit} className="profile-editor">
          <div className="section-caption"><span>HỒ SƠ ĐẦU VÀO</span><span>12 ĐẶC TRƯNG <SlidersHorizontal size={13} /></span></div>
          {model?.id === 'Simple' && <div className="variant-panel">
            <label htmlFor="simple-variant">Biến hồi quy đơn</label>
            <select id="simple-variant" value={variant} onChange={e => setVariant(e.target.value)} disabled={busy}>
              {Object.entries(data.simple_variants).map(([id, field]) =>
                <option key={id} value={id}>{id} · {data.fields.find(f => f.id === field)?.label}</option>)}
            </select>
            <small>Simple chỉ sử dụng một biến; các giá trị còn lại được giữ trong hồ sơ.</small>
          </div>}
          {groups.map((title, group) => <section className={`profile-group group-${group}`} key={title}>
            <div className="group-heading"><span className="group-number">0{group + 1}</span><div><h2>{title}</h2><p>{groupNotes[group]}</p></div></div>
            <div className="field-grid">{data.fields.filter(f => f.group === group).map(field =>
              <InputField key={field.id} field={field} value={profile[field.id]} disabled={busy}
                unused={!usedFields.includes(field.id)} onChange={value => setProfile(previous => ({ ...previous, [field.id]: value }))} />)}
            </div>
          </section>)}
          <p className="editor-note"><CircleHelp size={15} />Đổi mô hình không làm mất hồ sơ. Kết quả chỉ cập nhật khi bạn nhấn dự đoán.</p>
        </form>
        <aside className="output-column" aria-label="Kết quả dự đoán">
          <div className="section-caption"><span>KẾT QUẢ THÍ NGHIỆM</span><span><span className="small-dot" /> PYTHON MODELS</span></div>
          <div className={`prediction-card ${result ? 'has-result' : ''}`} aria-busy={pending}>
            {pending ? <div className="empty-result" role="status"><LoaderCircle className="spin" size={32} /><h2>Đang dự đoán…</h2><p>Đang chạy mô hình trên hồ sơ của bạn.</p></div> :
              result ? <><div className="result-top"><span>GPA DỰ ĐOÁN</span><span className="result-badge">RAW OUTPUT</span></div>
                <div className="gpa-value">{result.gpa.toFixed(3)}<span>GPA</span></div>
                <div className="result-bottom"><span data-testid="result-model">{result.model}{result.variant ? ` · ${result.variant}` : ''}</span><span>Ước lượng từ mô hình</span></div></> :
                <div className="empty-result"><span className="empty-icon"><Sparkles size={25} /></span><h2>Chưa có kết quả</h2><p>Hoàn tất hồ sơ và nhấn <strong>Dự đoán</strong><br />để khám phá GPA ước lượng.</p></div>}
          </div>
          {dirty && <p className="dirty-notice" role="status"><TriangleAlert size={17} /><span><strong>Đầu vào đã thay đổi — hãy dự đoán lại.</strong><br />Đang hiển thị kết quả lần dự đoán trước của {result?.model}.</span></p>}
          {error && <p role="alert" className="error-notice"><TriangleAlert size={17} />{error}</p>}
          {!model?.available && <p className="error-notice">Chưa có artifact cho mô hình này.</p>}
          <button form="profile-form" type="submit" className="predict-button" disabled={busy || !model?.available}>
            {pending ? <LoaderCircle size={18} className="spin" /> : <Sparkles size={18} />}
            {pending ? 'Đang dự đoán…' : cta}<ArrowUpRight size={18} />
          </button>
          <div className="chart-panel">
            <div className="chart-title"><div><p className="eyebrow">WHAT IF / ĐƯỜNG DỰ ĐOÁN</p><h2>Khi một biến thay đổi</h2></div><LineChart size={21} /></div>
            <label htmlFor="curve-variable">Biến khảo sát</label>
            <select id="curve-variable" value={selectedVariable} disabled={busy} onChange={e => setVariable(e.target.value)}>
              {variables.map(id => <option key={id} value={id}>{data.fields.find(f => f.id === id)?.label}</option>)}
            </select>
            {result ? <><SensitivityChart result={result} label={resultVariableLabel} />
              <div className="chart-legend"><span><i />Đường dự đoán</span><span><i className="dot" />Hồ sơ đã dự đoán</span></div>
              {dirty && <p className="chart-note">Biểu đồ của lần dự đoán trước · {resultVariableLabel}</p>}</> :
              <div className="chart-empty"><ChartNoAxesCombined size={36} /><p>Biểu đồ sẽ xuất hiện sau khi dự đoán</p></div>}
            <p className="chart-note">Các đầu vào khác giữ nguyên theo hồ sơ đã dự đoán. Đường này không biểu thị tác động nhân quả.</p>
          </div>
          {result?.out_of_range && <p className="dirty-notice"><TriangleAlert size={17} /><span>GPA ngoài miền 0–4. Giữ nguyên giá trị, <strong>không cắt (no clipping)</strong>.</span></p>}
          <footer className="scientific-note"><CircleHelp size={16} /><p>Dữ liệu sinh viên tổng hợp. Kết quả là ước lượng, không phải GPA thực tế và không chứng minh AI gây thay đổi GPA.</p></footer>
          <div className="output-signature"><span>LAB / 01</span><span>EXPERIMENT. OBSERVE. UNDERSTAND.</span></div>
        </aside>
      </div>}
    </main>
  </div>
}
