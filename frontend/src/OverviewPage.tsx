import { ArrowRight, ArrowUpRight, Check, CircleHelp, FlaskConical, Layers, MousePointer2, SlidersHorizontal } from 'lucide-react'
import type { Bootstrap, Page } from './types'
import { modelIcons } from './model-catalog'

export default function OverviewPage({ data, onNavigate, onExplore, busy }: {
  data: Bootstrap; onNavigate: (page: Page) => void; onExplore: (index: number) => void; busy: boolean
}) {
  return <div className="overview-page">
    <section className="overview-hero">
      <div className="hero-grid" aria-hidden="true" />
      <div className="hero-copy">
        <p className="eyebrow">INTERACTIVE MACHINE LEARNING LAB</p>
        <h1>Những con số<br /><span>kể chuyện GPA.</span></h1>
        <p>Một hồ sơ sinh viên. Sáu cách nhìn dữ liệu.<br />Khám phá dự đoán GPA qua các mô hình hồi quy.</p>
        <div className="hero-actions">
          <button className="hero-primary" onClick={() => onNavigate('prediction')}><FlaskConical size={17} />Mở phòng thí nghiệm<ArrowUpRight size={17} /></button>
          <button className="hero-secondary" onClick={() => onNavigate('comparison')}>Xem bảng so sánh<ArrowRight size={16} /></button>
        </div>
        <div className="hero-footnote"><span>KHÁM PHÁ DỮ LIỆU, KHÔNG SUY DIỄN NHÂN QUẢ</span><span>LAB / 01</span></div>
      </div>
      <div className="hero-visual" aria-hidden="true">
        <span className="visual-kicker">SIX PERSPECTIVES / ONE PROFILE</span>
        <svg viewBox="0 0 340 240">
          {[40, 80, 120, 160, 200].map(y => <line key={y} x1="10" x2="330" y1={y} y2={y} stroke="#ffffff12" />)}
          {['M10 190 Q160 170 330 50', 'M10 200 Q140 60 330 70', 'M10 175 L110 150 L210 100 L330 70',
            'M10 190 Q170 200 330 40', 'M10 180 L330 70', 'M10 195 L330 60'].map((d, i) =>
            <path key={d} d={d} fill="none" stroke={['#7eb5ff','#65d8c3','#efce77','#aaa0ff','#f2e99b','#eb9baf'][i]} strokeWidth="2" opacity=".8" />)}
          <circle cx="210" cy="100" r="5" fill="#f2e99b" />
        </svg>
        <span className="visual-disclaimer">Đường trang trí — không phải kết quả dự đoán</span>
      </div>
    </section>
    <div className="overview-stats">
      <div><strong>{String(data.models.filter(m => m.available).length).padStart(2, '0')}<span> / 06</span></strong><small>MÔ HÌNH SẴN SÀNG</small></div>
      <div><strong>{data.fields.length}</strong><small>ĐẶC TRƯNG ĐẦU VÀO</small></div>
      <div><strong>{data.validation_count?.toLocaleString('vi-VN') ?? '—'}</strong><small>DÒNG VALIDATION</small></div>
    </div>
    <section className="overview-section">
      <p className="eyebrow">LỘ TRÌNH / 01</p><h2>Khám phá trong ba bước.</h2>
      <div className="journey-grid">
        {[{ title: 'Hoàn thiện hồ sơ', text: 'Nhập 12 đặc trưng hoặc thử một hồ sơ mẫu.', Icon: SlidersHorizontal },
          { title: 'Chọn mô hình', text: 'Chuyển giữa sáu cách tiếp cận, giữ nguyên hồ sơ.', Icon: Layers },
          { title: 'Nhấn Dự đoán', text: 'Chủ động chạy mô hình và quan sát kết quả.', Icon: MousePointer2 }].map(({ title, text, Icon }, i) =>
          <div className="journey-step" key={title}><span className="journey-index">0{i + 1}</span><Icon size={20} /><h3>{title}</h3><p>{text}</p></div>)}
      </div>
    </section>
    <section className="overview-section">
      <div className="catalog-heading"><div><p className="eyebrow">BỘ SƯU TẬP / 02</p><h2>Hệ thống mô hình Lab 01.</h2></div><span>06 MODELS / MỘT CÂU HỎI</span></div>
      <div className="model-catalog">{data.models.map((model, i) => {
        const Icon = modelIcons[i]
        return <article className={`catalog-card catalog-color-${i}`} key={model.id}>
          <div className="catalog-card-top"><span className={`model-icon model-color-${i}`}><Icon size={21} /></span><span>0{i + 1} / MODEL STUDY</span></div>
          <h3>{model.id}</h3><p>{model.description}</p>
          <div className="catalog-meta"><span>{model.fields.length} đặc trưng</span><span><Check size={12} />{model.available ? 'Sẵn sàng' : 'Chưa có artifact'}</span></div>
          <button aria-label={`Khám phá ${model.id}`} disabled={busy} onClick={() => onExplore(i)}>Khám phá<ArrowUpRight size={16} /></button>
        </article>
      })}</div>
    </section>
    <footer className="scientific-note overview-note"><CircleHelp size={17} /><p>Dữ liệu sinh viên tổng hợp. Dự đoán là ước lượng, không phải GPA thực tế và không chứng minh AI gây thay đổi GPA.</p></footer>
  </div>
}
