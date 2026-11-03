
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="국가별 MBTI 분석",
    page_icon="🌍",
    layout="wide"
)

# 제목
st.title("🌍 국가별 MBTI 비율 분석")
st.write("국가를 선택하면 MBTI 유형별 비율을 그래프로 확인할 수 있습니다.")

# CSV 파일 불러오기
CSV_PATH = Path(__file__).parent / "countriesMBTI_16types.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)

    if "Country" not in df.columns:
        raise ValueError("Country 열이 없습니다.")

    mbti_columns = [
        col for col in df.columns if col != "Country"
    ]

    for col in mbti_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Country"])

    return df, mbti_columns


try:
    df, mbti_columns = load_data()

except FileNotFoundError:
    st.error("CSV 파일을 찾을 수 없습니다.")
    st.stop()

except Exception as e:
    st.error(f"데이터 오류: {e}")
    st.stop()


# 국가 선택
countries = sorted(df["Country"].astype(str).unique())

selected_country = st.selectbox(
    "국가를 선택하세요",
    countries
)

# 선택한 국가 데이터
country_data = df[
    df["Country"].astype(str) == selected_country
].iloc[0]


# MBTI 데이터 정리
chart_df = pd.DataFrame({
    "MBTI": mbti_columns,
    "비율": [
        country_data[col]
        for col in mbti_columns
    ]
})

chart_df = chart_df.dropna()

# 소수 형태를 퍼센트로 변환
if chart_df["비율"].max() <= 1:
    chart_df["비율(%)"] = chart_df["비율"] * 100
else:
    chart_df["비율(%)"] = chart_df["비율"]

# 비율 높은 순서대로 정렬
chart_df = chart_df.sort_values(
    "비율(%)",
    ascending=False
).reset_index(drop=True)


# 그래프 색상 설정
blue_shades = [
    "#1D4ED8",
    "#3267DC",
    "#477FE0",
    "#5C97E5",
    "#71AFE9",
    "#86C2ED",
    "#9BCFF1",
    "#B0DCF5",
    "#C5E8F9",
    "#D9F1FC",
    "#EAF8FE",
    "#F2FAFF",
    "#DCEEFF",
    "#C9E4FF",
    "#B6DAFF"
]

# 1등 빨간색, 나머지 파란색 그라데이션
bar_colors = (
    ["#E53935"]
    + blue_shades[:len(chart_df) - 1]
)


# 최고 유형 표시
st.subheader(
    f"📊 {selected_country}의 MBTI 비율"
)

top_type = chart_df.iloc[0]

col1, col2 = st.columns(2)

col1.metric(
    "가장 높은 MBTI",
    top_type["MBTI"]
)

col2.metric(
    "1위 비율",
    f'{top_type["비율(%)"]:.2f}%'
)


# 막대그래프 만들기
fig = go.Figure(
    data=[
        go.Bar(
            x=chart_df["MBTI"],
            y=chart_df["비율(%)"],

            marker=dict(
                color=bar_colors
            ),

            text=[
                f"{v:.2f}%"
                for v in chart_df["비율(%)"]
            ],

            textposition="outside",

            hovertemplate=(
                "<b>%{x}</b><br>"
                "비율: %{y:.2f}%"
                "<extra></extra>"
            )
        )
    ]
)


# 그래프 디자인
fig.update_layout(
    xaxis_title="MBTI 유형",
    yaxis_title="비율(%)",

    yaxis=dict(
        rangemode="tozero",
        ticksuffix="%"
    ),

    showlegend=False,
    height=560,

    plot_bgcolor="white",
    paper_bgcolor="white"
)


# 화면에 그래프 출력
st.plotly_chart(
    fig,
    use_container_width=True
)


# 데이터 표
with st.expander("MBTI 상세 비율 보기"):

    display_df = chart_df[
        ["MBTI", "비율(%)"]
    ].copy()

    display_df["비율(%)"] = (
        display_df["비율(%)"]
        .map(lambda x: f"{x:.2f}%")
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# 안내 문구
st.caption(
    "본 자료는 제공된 CSV 데이터에 기반한 분석 결과입니다."
)
