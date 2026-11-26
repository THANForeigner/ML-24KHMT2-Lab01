import altair as alt
import pandas as pd


def sensitivity(frame, label, current):
    base = alt.Chart(frame).encode(
        x=alt.X("x:Q", title=label, scale=alt.Scale(zero=False)),
        y=alt.Y("GPA:Q", title="GPA dự đoán", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("x:Q", title=label, format=".2f"),
                 alt.Tooltip("GPA:Q", title="GPA dự đoán", format=".3f")],
    )
    line = base.mark_line(color="#11665e", strokeWidth=3)
    current_point = base.transform_filter(alt.datum.x == current).mark_point(
        size=165, filled=True, color="#b45b20", stroke="#fffefa", strokeWidth=2
    )
    return line + current_point


def agreement(points):
    sample = points.sample(min(800, len(points)), random_state=42)
    lo, hi = points.min().min(), points.max().max()
    diagonal = pd.DataFrame({"Thực tế": [lo, hi], "Dự đoán": [lo, hi]})
    axes = dict(x=alt.X("Thực tế:Q", scale=alt.Scale(domain=[lo, hi], zero=False)),
                y=alt.Y("Dự đoán:Q", scale=alt.Scale(domain=[lo, hi], zero=False)))
    return (alt.Chart(sample).mark_circle(color="#11665e", opacity=.35, size=55)
            .encode(**axes, tooltip=[alt.Tooltip("Thực tế:Q", format=".3f"),
                                     alt.Tooltip("Dự đoán:Q", format=".3f")])
            + alt.Chart(diagonal).mark_line(color="#b45b20", strokeDash=[5, 5], strokeWidth=2)
            .encode(**axes))
