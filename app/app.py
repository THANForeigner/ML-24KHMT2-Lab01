import streamlit as st

from pages import model_comparison, prediction
from services.predictor import MODELS, dataset, model
from theme import apply, eyebrow


def showcase():
    apply()
    st.sidebar.markdown('<div class="sidebar-brand"><span class="brand-mark" aria-hidden="true">L</span>'
                        '<div><strong>LAB / 01</strong><small>MODEL EXPLORER</small></div></div>',
                        unsafe_allow_html=True)
    section = st.sidebar.radio("Khám phá", ["Tổng quan", "Thử dự đoán", "So sánh"], key="navigation")
    st.sidebar.divider()
    st.sidebar.caption("HỒI QUY TUYẾN TÍNH & MỞ RỘNG")
    st.sidebar.markdown('<p class="sidebar-note">Một phòng thí nghiệm nhỏ để khám phá '
                        'dự đoán GPA từ dữ liệu sinh viên tổng hợp.</p>', unsafe_allow_html=True)
    train = dataset("train")
    if "profile" not in st.session_state:
        st.session_state.profile = train.dropna().iloc[0].to_dict()
    available = []
    for name in MODELS:
        try:
            model(name)
            available.append(name)
        except Exception as exc:
            st.sidebar.warning(f"{name} chưa sẵn sàng: {exc}")
    if section == "Tổng quan":
        st.markdown(
            '<div class="hero">'
            '<div class="hero-grid" aria-hidden="true"></div>'
            '<div class="hero-content">'
            '<div class="hero-kicker"><span class="status-dot"></span> INTERACTIVE MACHINE LEARNING LAB</div>'
            '<h1>Những con số<br><em>kể chuyện GPA.</em></h1>'
            '<p>Thử một hồ sơ sinh viên. Đặt bốn thuật toán cạnh nhau. '
            'Khám phá cách mô hình phản ứng khi bạn thay đổi từng đầu vào.</p>'
            '<div class="hero-line" aria-hidden="true"></div>'
            '<div class="hero-bottom"><span>KHÁM PHÁ DỮ LIỆU, KHÔNG SUY DIỄN NHÂN QUẢ</span>'
            '<span>01 / 03</span></div></div></div>', unsafe_allow_html=True
        )
        st.markdown('<div class="stats-strip">'
                    f'<div><strong>{len(available):02d}<span> / 04</span></strong><small>MÔ HÌNH SẴN SÀNG</small></div>'
                    '<div><strong>12</strong><small>ĐẶC TRƯNG ĐẦU VÀO</small></div>'
                    '<div><strong>7.500</strong><small>DÒNG VALIDATION</small></div>'
                    '</div>', unsafe_allow_html=True)
        st.write("")
        eyebrow("Lộ trình / 01")
        st.subheader("Bắt đầu từ một hồ sơ")
        st.write("Điền dữ liệu mẫu để thử ngay, hoặc nhập thông tin của riêng bạn. "
                 "Các mô hình sẽ dùng cùng hồ sơ này khi so sánh.")
        left, right = st.columns(2, gap="medium")
        with left:
            with st.container(border=True):
                st.markdown('<span class="card-index">01 — EXPLORE</span>', unsafe_allow_html=True)
                st.subheader("Thử dự đoán")
                st.write("Điều chỉnh từng thông số và theo dõi đường GPA dự đoán thay đổi.")
                st.button("Mở phòng thí nghiệm →", type="primary", use_container_width=True,
                          key="start_prediction", on_click=lambda: st.session_state.update(navigation="Thử dự đoán"))
        with right:
            with st.container(border=True):
                st.markdown('<span class="card-index">02 — COMPARE</span>', unsafe_allow_html=True)
                st.subheader("So sánh thuật toán")
                st.write("Đối chiếu dự đoán và hiệu năng trên cùng tập validation.")
                st.button("Xem bảng so sánh →", use_container_width=True, key="start_comparison",
                          on_click=lambda: st.session_state.update(navigation="So sánh"))
        st.write("")
        eyebrow("Bộ sưu tập / 02")
        st.subheader("Chọn mô hình để khám phá")
        st.write("Bốn cách tiếp cận một câu hỏi: GPA sau học kỳ sẽ được ước lượng như thế nào?")
        grid = st.columns(2, gap="medium")
        for i, (name, (_, fields, text)) in enumerate(MODELS.items()):
            with grid[i % 2]:
                with st.container(border=True):
                    st.markdown(f'<span class="card-index">0{i + 1} / MODEL STUDY</span>', unsafe_allow_html=True)
                    st.subheader(name)
                    st.write(text)
                    st.caption(f"{len(fields)} đặc trưng  ·  "
                               + ("Sẵn sàng khám phá" if name in available else "Chưa tải được"))
        st.info("Lưu ý: Đây là dữ liệu tổng hợp. Dự đoán không chứng minh AI gây thay đổi GPA.")
    elif not available:
        st.error("Chưa có pipeline sẵn sàng. Kiểm tra thư mục outputs/ và môi trường Python.")
    elif section == "Thử dự đoán":
        prediction.render(train, available)
    else:
        model_comparison.render(available)


st.set_page_config(page_title="Lab 01 · Hồi quy GPA", layout="wide")
st.navigation([st.Page(showcase, title="Showcase GPA")], position="hidden").run()
