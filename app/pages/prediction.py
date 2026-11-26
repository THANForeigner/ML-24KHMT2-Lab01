import numpy as np
import pandas as pd
import streamlit as st

from components.charts import sensitivity
from services.predictor import MODELS, SIMPLE_FEATURES, predict
from theme import eyebrow

LABEL = {
    "Pre_Semester_GPA": "GPA trước học kỳ", "Traditional_Study_Hours": "Giờ tự học truyền thống",
    "Year_of_Study": "Năm học", "Major_Category": "Nhóm ngành",
    "Weekly_GenAI_Hours": "Giờ dùng AI mỗi tuần", "Primary_Use_Case": "Mục đích dùng AI",
    "Prompt_Engineering_Skill": "Kỹ năng viết prompt", "Tool_Diversity": "Số công cụ",
    "Paid_Subscription": "Có thuê bao trả phí", "Institutional_Policy": "Quy định của trường",
    "Perceived_AI_Dependency": "Mức phụ thuộc AI", "Anxiety_Level_During_Exams": "Mức lo âu khi thi",
}

MODEL_LABEL = {
    "Simple": "Hồi quy tuyến tính đơn · Simple",
    "Interaction": "Hồi quy có tương tác · Interaction",
    "Piecewise": "Hồi quy tuyến tính từng đoạn · Piecewise",
    "Polynomial": "Hồi quy đa thức · Polynomial",
    "Ridge": "Hồi quy Ridge · Ridge",
    "Elastic Net": "Hồi quy Elastic Net · Elastic Net",
}
OPTIONS_VI = {
    "Year_of_Study": {
        "Freshman": "Năm nhất", "Sophomore": "Năm hai", "Junior": "Năm ba",
        "Senior": "Năm tư", "Graduate": "Sau đại học",
    },
    "Major_Category": {
        "Arts": "Nghệ thuật", "Business": "Kinh doanh", "Humanities": "Khoa học nhân văn",
        "Medical": "Y khoa", "STEM": "Khoa học, công nghệ, kỹ thuật & toán",
    },
    "Primary_Use_Case": {
        "Copywriting/Drafting": "Soạn thảo nội dung",
        "Debugging/Troubleshooting": "Gỡ lỗi và khắc phục sự cố",
        "Direct_Answer_Generation": "Tạo câu trả lời trực tiếp",
        "Ideation": "Lên ý tưởng", "Summarizing_Reading": "Tóm tắt tài liệu",
    },
    "Prompt_Engineering_Skill": {
        "Beginner": "Mới bắt đầu", "Intermediate": "Trung cấp", "Advanced": "Nâng cao",
    },
    "Institutional_Policy": {
        "Strict_Ban": "Cấm sử dụng", "Allowed_With_Citation": "Cho phép nếu trích dẫn",
        "Actively_Encouraged": "Khuyến khích sử dụng",
    },
    "Paid_Subscription": {False: "Không đăng ký", True: "Có đăng ký"},
}


def option_label(field, value):
    if field in OPTIONS_VI:
        return OPTIONS_VI[field][value]
    if field in ("Perceived_AI_Dependency", "Anxiety_Level_During_Exams"):
        return f"Mức {value}/10"
    if field == "Tool_Diversity":
        return f"{value} công cụ"
    return str(value)


def load_sample(train, state):
    """Synchronize both the prediction profile and Streamlit's widget values."""
    samples = train.dropna()
    index = (state.get("sample_index", 0) + 1) % len(samples)
    state["sample_index"] = index
    profile = samples.iloc[index].to_dict()
    state["profile"] = profile
    for field, value in profile.items():
        state["field_" + field] = value


def remember_simple_variant():
    st.session_state.simple_selection = st.session_state.simple_variant


def render(train, available):
    eyebrow("Phòng thí nghiệm / 01")
    st.title("Một hồ sơ. Nhiều khả năng.")
    st.write("Tạo hồ sơ sinh viên và quan sát cách mô hình ước lượng GPA sau học kỳ.")
    head, sample = st.columns([2, 1], vertical_alignment="bottom", gap="medium")
    with head:
        name = st.selectbox("Mô hình dự đoán", available, format_func=MODEL_LABEL.get, key="model")
    _, fields, description = MODELS[name]
    with sample:
        st.button("Điền dữ liệu mẫu", use_container_width=True, on_click=load_sample,
                  args=(train, st.session_state))
    st.info(f"**{name}** · {description}")
    variant = "M03"
    if name == "Simple":
        variant = st.selectbox(
            "Chọn biến dùng cho hồi quy đơn",
            list(SIMPLE_FEATURES),
            index=list(SIMPLE_FEATURES).index(st.session_state.get("simple_selection", "M03")),
            format_func=lambda key: f"{key} · {LABEL[SIMPLE_FEATURES[key]]}",
            key="simple_variant",
            on_change=remember_simple_variant,
        )
        fields = [SIMPLE_FEATURES[variant]]
        st.caption(f"Chỉ {LABEL[fields[0]].lower()} được dùng để dự đoán với {variant}. "
                   "Đổi lựa chọn để thử một model Simple khác; tất cả đều không clipping.")
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
        if not group_fields:
            continue
        with st.container(border=True):
            st.markdown(f'<span class="card-index">{number} / INPUT</span>', unsafe_allow_html=True)
            st.subheader(title)
            st.caption(detail)
            columns = st.columns(2)
            for i, field in enumerate(group_fields):
                values, widget = train[field], columns[i % 2]
                key = "field_" + field
                if field in ("Weekly_GenAI_Hours", "Traditional_Study_Hours", "Pre_Semester_GPA"):
                    initial = {} if key in st.session_state else {"value": float(profile[field])}
                    profile[field] = widget.number_input(LABEL[field], float(values.min()), float(values.max()),
                                                         step=.001, format="%.3f", key=key, **initial)
                else:
                    options = sorted(values.dropna().unique().tolist())
                    initial = {} if key in st.session_state else {"index": options.index(profile[field])}
                    profile[field] = widget.selectbox(
                        LABEL[field], options, **initial,
                        format_func=lambda value, column=field: option_label(column, value), key=key
                    )
    st.session_state.profile = profile
    frame = pd.DataFrame([profile])
    st.write("")
    eyebrow("Phản hồi / theo thời gian thực")
    st.subheader("Kết quả dự đoán")
    prediction = float(predict(name, frame, variant=variant)[0])
    st.metric(f"GPA dự đoán · {name}" + (f" {variant}" if name == "Simple" else ""),
              f"{prediction:.3f}", border=True)
    if not 0 <= prediction <= 4:
        st.warning("Dự đoán nằm ngoài miền GPA 0–4. Web giữ nguyên giá trị, không cắt (no clipping).")
    st.caption("Giá trị ước lượng từ mô hình, không phải GPA thực tế của sinh viên.")
    with st.container(border=True):
        st.markdown('<span class="card-index">WHAT IF / ĐƯỜNG DỰ ĐOÁN</span>', unsafe_allow_html=True)
        st.subheader("Điều gì xảy ra khi thay đổi một biến?")
        st.write("Các trường khác giữ nguyên theo hồ sơ đang hiển thị ở trên.")
        candidates = ["Weekly_GenAI_Hours", "Pre_Semester_GPA", "Traditional_Study_Hours",
                      "Tool_Diversity", "Perceived_AI_Dependency"]
        field = st.selectbox("Biến khảo sát", [column for column in candidates if column in fields],
                             format_func=LABEL.get)
        if field in ("Tool_Diversity", "Perceived_AI_Dependency"):
            grid = np.sort(train[field].dropna().unique())
        else:
            grid = np.unique(np.r_[np.linspace(train[field].min(), train[field].max(), 61), profile[field]])
        rows = pd.DataFrame([profile] * len(grid)).assign(**{field: grid})
        chart = pd.DataFrame({"x": grid, "GPA": predict(name, rows, variant=variant)})
        st.altair_chart(sensitivity(chart, LABEL[field], profile[field]), width="stretch")
        st.caption("Chấm cam là hồ sơ hiện tại. Các đầu vào khác giữ nguyên; đường này không biểu thị tác động nhân quả.")
