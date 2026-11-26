from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


def test_overview_has_clear_entry_points_and_prediction_groups():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/app.py"),
                            default_timeout=45).run()
    assert not app.exception
    assert any("Chọn mô hình" in heading.value for heading in app.subheader)
    app.button(key="start_prediction").click().run()
    assert not app.exception
    assert app.selectbox(key="model")
    assert any("Thông tin học tập" in heading.value for heading in app.subheader)
    assert any("Kết quả dự đoán" in heading.value for heading in app.subheader)


def test_overview_comparison_shortcut_shows_validation_context():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/app.py"),
                            default_timeout=45).run()
    assert any("Bắt đầu" in heading.value for heading in app.subheader)
    app.button(key="start_comparison").click().run()
    assert not app.exception
    assert any("Hiệu năng trên tập validation" in heading.value for heading in app.subheader)
    assert app.dataframe[1].value.shape[0] == 4


def test_sample_button_updates_visible_fields_and_prediction_together():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/app.py"),
                            default_timeout=45).run()
    app.radio[0].set_value("Thử dự đoán").run()
    app.number_input(key="field_Pre_Semester_GPA").set_value(3.5).run()
    app.number_input(key="field_Weekly_GenAI_Hours").set_value(2.0).run()
    before_gpa = app.metric[0].value
    app.button[0].click().run()
    assert not app.exception
    shown = app.number_input(key="field_Pre_Semester_GPA").value
    assert shown == pytest.approx(float(app.session_state["profile"]["Pre_Semester_GPA"]))
    assert app.metric[0].value != before_gpa
    assert shown != 3.5
    assert app.number_input(key="field_Weekly_GenAI_Hours").value != 2.0
    assert app.selectbox(key="field_Major_Category").value == app.session_state["profile"]["Major_Category"]
    app.button[0].click().run()
    assert app.number_input(key="field_Weekly_GenAI_Hours").value == pytest.approx(
        float(app.session_state["profile"]["Weekly_GenAI_Hours"]))


def test_sample_selection_sets_existing_widget_state_not_just_profile():
    import pandas as pd
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
    from pages.prediction import load_sample

    rows = pd.DataFrame({"Pre_Semester_GPA": [2.1, 3.8], "Weekly_GenAI_Hours": [4.0, 7.0]})
    state = {"profile": rows.iloc[0].to_dict(), "sample_index": 0,
             "field_Pre_Semester_GPA": 2.1, "field_Weekly_GenAI_Hours": 4.0}
    load_sample(rows, state)
    assert state["field_Pre_Semester_GPA"] == 3.8
    assert state["field_Weekly_GenAI_Hours"] == 7.0
    assert state["profile"]["Pre_Semester_GPA"] == 3.8


def test_demo_predicts_and_compares_same_profile():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/app.py"),
                            default_timeout=45).run()
    assert app.radio
    app.radio[0].set_value("Thử dự đoán").run()
    for name, expected in [("Interaction", "2.699"), ("Ridge", "2.648"),
                           ("Polynomial", "2.562"), ("Elastic Net", "2.685")]:
        app.selectbox(key="model").select(name).run()
        assert not app.exception
        assert app.metric[0].value == expected
    app.number_input(key="field_Pre_Semester_GPA").set_value(3.5).run()
    changed = app.metric[0].value
    assert changed != "2.685"
    app.radio[0].set_value("So sánh").run()
    assert not app.exception
    assert any("Cùng một hồ sơ" in heading.value for heading in app.subheader)
    assert len(app.get("vega_lite_chart")) >= 2
    scores = app.dataframe[0].value.set_index("Thuật toán")
    assert scores.loc["Elastic Net", "GPA"] == pytest.approx(float(changed), abs=.0005)
    metrics = app.dataframe[1].value.set_index("Thuật toán")
    assert metrics.loc["Ridge", "RMSE"] == pytest.approx(.158855, abs=.00001)
    assert metrics.loc["Elastic Net", "RMSE"] == pytest.approx(.150078, abs=.00001)
