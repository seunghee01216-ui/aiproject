
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

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path


# =========================
# 1. 페이지 설정
# =========================

st.set_page_config(
    page_title="국가별 MBTI 분석",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 국가별 MBTI 분석 프로그램")

st.write(
    "국가별 MBTI 비율과 MBTI 유형별 국가 순위를 확인할 수 있습니다."
)


# =========================
# 2. CSV 데이터 불러오기
# =========================

CSV_PATH = Path(__file__).parent / "countriesMBTI_16types.csv"


@st.cache_data
def load_data():

    df = pd.read_csv(CSV_PATH)

    if "Country" not in df.columns:
        raise ValueError("Country 열이 없습니다.")

    mbti_columns = [
        col for col in df.columns
        if col != "Country"
    ]

    for col in mbti_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

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


# =========================
# 3. 색상 그라데이션 함수
# =========================

def make_colors(count):

    if count <= 0:
        return []

    colors = ["#E53935"]

    # 1위는 빨간색
    # 나머지는 진한 파란색에서 연한 파란색
    start = (29, 78, 216)
    end = (234, 248, 254)

    for i in range(count - 1):

        if count <= 2:
            ratio = 0
        else:
            ratio = i / (count - 2)

        r = int(start[0] + (end[0] - start[0]) * ratio)
        g = int(start[1] + (end[1] - start[1]) * ratio)
        b = int(start[2] + (end[2] - start[2]) * ratio)

        colors.append(f"#{r:02X}{g:02X}{b:02X}")

    return colors


# =========================
# 4. 그래프 함수
# =========================

def draw_chart(data, x_col, y_col, title, x_title):

    colors = make_colors(len(data))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=data[x_col],
            y=data[y_col],

            marker=dict(
                color=colors
            ),

            text=[
                f"{v:.2f}%"
                for v in data[x_col]
            ],

            textposition="outside",

            hovertemplate=(
                "<b>%{y}</b><br>"
                "비율: %{x:.2f}%"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="비율(%)",
        yaxis_title=x_title,
        xaxis=dict(
            rangemode="tozero",
            ticksuffix="%"
        ),
        showlegend=False,
        height=max(550, len(data) * 28),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=30, r=50, t=60, b=40)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================
# 5. 탭 만들기
# =========================

tab1, tab2 = st.tabs([
    "🌍 국가별 MBTI 분석",
    "🧠 MBTI별 국가 순위"
])


# =========================
# 6. 국가별 MBTI 분석
# =========================

with tab1:

    st.header("국가를 선택하세요")

    countries = sorted(
        df["Country"].astype(str).unique()
    )

    selected_country = st.selectbox(
        "국가 선택",
        countries,
        key="country_select"
    )

    country_data = df[
        df["Country"].astype(str) == selected_country
    ].iloc[0]

    chart_df = pd.DataFrame({
        "MBTI": mbti_columns,
        "비율": [
            country_data[col]
            for col in mbti_columns
        ]
    }).dropna()

    if not chart_df.empty:

        if chart_df["비율"].max() <= 1:
            chart_df["비율(%)"] = chart_df["비율"] * 100
        else:
            chart_df["비율(%)"] = chart_df["비율"]

        chart_df = chart_df.sort_values(
            "비율(%)",
            ascending=False
        ).reset_index(drop=True)

        st.subheader(
            f"📊 {selected_country}의 MBTI 비율"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "가장 높은 MBTI",
            chart_df.iloc[0]["MBTI"]
        )

        col2.metric(
            "1위 비율",
            f'{chart_df.iloc[0]["비율(%)"]:.2f}%'
        )

        draw_chart(
            chart_df,
            "비율(%)",
            "MBTI",
            f"{selected_country} MBTI 순위",
            "MBTI 유형"
        )

        with st.expander("상세 데이터 보기"):

            st.dataframe(
                chart_df[["MBTI", "비율(%)"]],
                use_container_width=True,
                hide_index=True
            )


# =========================
# 7. MBTI별 국가 순위
# =========================

with tab2:

    st.header("MBTI 유형을 선택하세요")

    selected_mbti = st.selectbox(
        "MBTI 선택",
        mbti_columns,
        key="mbti_select"
    )

    st.subheader(
        f"🏆 {selected_mbti} 비율이 높은 국가 순위"
    )

    ranking_df = df[
        ["Country", selected_mbti]
    ].copy()

    ranking_df = ranking_df.dropna(
        subset=[selected_mbti]
    )

    # 백분율 변환
    if not ranking_df.empty:

        if ranking_df[selected_mbti].max() <= 1:
            ranking_df["비율(%)"] = (
                ranking_df[selected_mbti] * 100
            )
        else:
            ranking_df["비율(%)"] = (
                ranking_df[selected_mbti]
            )

        # 비율이 높은 국가부터 정렬
        ranking_df = ranking_df.sort_values(
            "비율(%)",
            ascending=False
        ).reset_index(drop=True)

        # 순위 추가
        ranking_df.insert(
            0,
            "순위",
            range(1, len(ranking_df) + 1)
        )

        # 1위 국가 정보
        first_country = ranking_df.iloc[0]

        col1, col2 = st.columns(2)

        col1.metric(
            "🥇 1위 국가",
            first_country["Country"]
        )

        col2.metric(
            "1위 비율",
            f'{first_country["비율(%)"]:.2f}%'
        )

        # 그래프 표시 개수 선택
        show_count = st.slider(
            "그래프에 표시할 국가 수",
            min_value=5,
            max_value=len(ranking_df),
            value=min(20, len(ranking_df)),
            key="show_count"
        )

        # 상위 국가만 그래프에 표시
        top_countries = ranking_df.head(
            show_count
        ).copy()

        # 그래프
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=top_countries["비율(%)"],
                y=top_countries["Country"],

                orientation="h",

                marker=dict(
                    color=make_colors(len(top_countries)),
                    line=dict(width=0)
                ),

                text=[
                    f"{v:.2f}%"
                    for v in top_countries["비율(%)"]
                ],

                textposition="outside",

                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "비율: %{x:.2f}%"
                    "<extra></extra>"
                )
            )
        )

        fig.update_layout(
            title=f"{selected_mbti} 국가별 비율 순위",
            xaxis_title="비율(%)",
            yaxis_title="국가",
            xaxis=dict(
                rangemode="tozero",
                ticksuffix="%"
            ),
            yaxis=dict(
                autorange="reversed"
            ),
            showlegend=False,
            height=max(550, show_count * 28),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # 전체 국가 순위 표
        st.subheader("📋 전체 국가 순위")

        display_df = ranking_df[
            ["순위", "Country", "비율(%)"]
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

    else:
        st.warning("선택한 MBTI의 데이터가 없습니다.")


# =========================
# 8. 하단 안내
# =========================

st.divider()

st.caption(
    "본 프로그램은 제공된 CSV 데이터에 기반한 분석 결과입니다. "
    "자료의 조사 방법과 표본에 따라 실제 인구 분포와 차이가 있을 수 있습니다."
)
