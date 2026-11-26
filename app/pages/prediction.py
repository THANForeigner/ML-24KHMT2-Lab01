import numpy as np
import pandas as pd
import streamlit as st

from components.charts import sensitivity
from services.predictor import MODELS, predict
from theme import eyebrow

LABEL = {
    "Pre_Semester_GPA": "GPA trước học kỳ", "Traditional_Study_Hours": "Giờ tự học truyền thống",
    "Year_of_Study": "Năm học", "Major_Category": "Nhóm ngành",
    "Weekly_GenAI_Hours": "Giờ dùng AI mỗi tuần", "Primary_Use_Case": "Mục đích dùng AI",
    "Prompt_Engineering_Skill": "Kỹ năng viết prompt", "Tool_Diversity": "Số công cụ",
    "Paid_Subscription": "Có thuê bao trả phí", "Institutional_Policy": "Quy định của trường",
    "Perceived_AI_Dependency": "Mức phụ thuộc AI", "Anxiety_Level_During_Exams": "Mức lo âu khi thi",
}


def load_sample(train, state):
    """Synchronize both the prediction profile and Streamlit's widget values."""
    samples = train.dropna()
    index = (state.get("sample_index", 0) + 1) % len(samples)
    state["sample_index"] = index
    profile = samples.iloc[index].to_dict()
    state["profile"] = profile
    for field, value in profile.items():
        state["field_" + field] = value


def render(train, available):
    eyebrow("Phòng thí nghiệm / 01")
    st.title("Một hồ sơ. Nhiều khả năng.")
    st.write("Tạo hồ sơ sinh viên và quan sát cách mô hình ước lượng GPA sau học kỳ.")
    head, sample = st.columns([2, 1], vertical_alignment="bottom", gap="medium")
    with head:
        name = st.selectbox("Mô hình dự đoán", available, key="model")
    _, fields, description = MODELS[name]
    with sample:
        st.button("Điền dữ liệu mẫu", use_container_width=True, on_click=load_sample,
                  args=(train, st.session_state))
    st.info(f"**{name}** · {description}")
    profile = st.session_state.profile.copy()
    groups = [
        ("01", "Thông tin học tập", "Nền tảng học tập trước kỳ dự đoán.",
         ["Pre_Semester_GPA", "Traditional_Study_Hours", "Year_of_Study", "Major_Category"]),
        ("02", "Thói quen sử dụng AI", "Cách thức và tần suất tương tác với công cụ AI.",
         ["Weekly_GenAI_Hours", "Primary_Use_Case",
          "Prompt_Engineering_Skill", "Tool_Diversity", "Paid_Subscription"]),
        ("03", "Bối cảnh cá nhân", "Những yếu tố còn lại trong hồ sơ của mô hình.",
         ["Institutional_Policy", "Perceived_AI_Dependency", "Anxiety_Level_During_Exams"]),
    ]
    eyebrow("Thiết lập / hồ sơ đầu vào")
    for number, title, detail, members in groups:
        group_fields = [field for field in members if field in fields]
        with st.container(border=True):
            st.markdown(f'<span class="card-index">{number} / INPUT</span>', unsafe_allow_html=True)
            st.subheader(title)
            st.caption(detail)
            columns = st.columns(2)
            for i, field in enumerate(group_fields):
                values, widget = train[field], columns[i % 2]
                key = "field_" + field
                if field in ("Weekly_GenAI_Hours", "Traditional_Study_Hours", "Pre_Semester_GPA"):
                    profile[field] = widget.number_input(LABEL[field], float(values.min()), float(values.max()),
                                                         float(profile[field]), step=.001, format="%.3f", key=key)
                else:
                    options = sorted(values.dropna().unique().tolist())
                    profile[field] = widget.selectbox(LABEL[field], options, index=options.index(profile[field]), key=key)
    st.session_state.profile = profile
    frame = pd.DataFrame([profile])
    st.write("")
    eyebrow("Phản hồi / theo thời gian thực")
    st.subheader("Kết quả dự đoán")
    st.metric(f"GPA dự đoán · {name}", f"{predict(name, frame)[0]:.3f}", border=True)
    st.caption("Giá trị ước lượng từ mô hình, không phải GPA thực tế của sinh viên.")
    with st.container(border=True):
        st.markdown('<span class="card-index">WHAT IF / ĐƯỜNG DỰ ĐOÁN</span>', unsafe_allow_html=True)
        st.subheader("Điều gì xảy ra khi thay đổi một biến?")
        st.write("Các trường khác giữ nguyên theo hồ sơ đang hiển thị ở trên.")
        field = st.selectbox("Biến khảo sát", ["Weekly_GenAI_Hours", "Pre_Semester_GPA", "Traditional_Study_Hours"],
                             format_func=LABEL.get)
        grid = np.unique(np.r_[np.linspace(train[field].min(), train[field].max(), 61), profile[field]])
        rows = pd.DataFrame([profile] * len(grid)).assign(**{field: grid})
        chart = pd.DataFrame({"x": grid, "GPA": predict(name, rows)})
        st.altair_chart(sensitivity(chart, LABEL[field], profile[field]), width="stretch")
        st.caption("Chấm cam là hồ sơ hiện tại. Các đầu vào khác giữ nguyên; đường này không biểu thị tác động nhân quả.")
