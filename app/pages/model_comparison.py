import pandas as pd
import streamlit as st
import altair as alt

from components.charts import agreement
from services.predictor import predict, validation
from theme import eyebrow


def render(available):
    eyebrow("Bảng đối chiếu / 02")
    st.title("Đặt các mô hình cạnh nhau.")
    st.write("So sánh kết quả của cùng một hồ sơ, sau đó kiểm tra sai số trên tập validation.")
    with st.container(border=True):
        st.markdown('<span class="card-index">01 / CÙNG ĐẦU VÀO</span>', unsafe_allow_html=True)
        st.subheader(f"Cùng một hồ sơ, {len(available)} góc nhìn")
        st.caption("Dùng hồ sơ hiện tại từ trang Thử dự đoán, hoặc dòng mẫu đầu tiên nếu bạn chưa nhập.")
        frame = pd.DataFrame([st.session_state.profile])
        scores = pd.DataFrame([{"Thuật toán": name, "GPA": predict(name, frame)[0]} for name in available])
        domain = [min(0, float(scores["GPA"].min())), max(4, float(scores["GPA"].max()))]
        bars = (alt.Chart(scores).mark_bar(color="#11665e", cornerRadiusEnd=5).encode(
            x=alt.X("GPA:Q", title="GPA dự đoán", scale=alt.Scale(domain=domain)),
            y=alt.Y("Thuật toán:N", sort=list(available), title=None),
            tooltip=["Thuật toán:N", alt.Tooltip("GPA:Q", format=".3f")]))
        st.altair_chart(bars, width="stretch")
        st.dataframe(scores, hide_index=True)
    st.write("")
    eyebrow("Đánh giá / 7.500 dòng")
    st.subheader("Hiệu năng trên tập validation")
    st.caption("Mô hình chỉ được đánh giá trên validation khi artifact không fit trên chính tập đó. "
               "Tập test không tham gia so sánh.")
    metrics, points = [], {}
    for name in available:
        try:
            row, points[name] = validation(name)
            metrics.append(row)
        except ValueError as exc:
            st.warning(f"{name}: {exc}")
    if points:
        with st.container(border=True):
            st.markdown('<span class="card-index">02 / CHỈ SỐ</span>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(metrics), hide_index=True)
            st.caption("MAE, RMSE: càng thấp càng tốt. R²: càng cao càng tốt.")
        with st.container(border=True):
            st.markdown('<span class="card-index">03 / ĐỘ LỆCH</span>', unsafe_allow_html=True)
            st.subheader("Thực tế so với dự đoán")
            name = st.selectbox("Chọn thuật toán trên biểu đồ", list(points))
            st.altair_chart(agreement(points[name]), width="stretch")
            st.caption(f"Chỉ số dùng đủ {len(points[name]):,} dòng validation; hình hiển thị tối đa 800 dòng. "
                       "Đường chéo nét đứt là dự đoán hoàn hảo.")
