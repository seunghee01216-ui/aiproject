
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="국가별 MBTI 분석",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 국가별 MBTI 분석 프로그램")

st.write(
    "국가별 MBTI 비율과 MBTI 유형별 국가 순위를 확인할 수 있습니다."
)


# =====================================
# 2. CSV 파일 경로 설정
# =====================================

# 현재 파일: pages/mbti_analysis.py
# 상위 폴더: mbti_project/
# CSV 파일: mbti_project/countriesMBTI_16types.csv

CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "countriesMBTI_16types.csv"
)


# =====================================
# 3. 데이터 불러오기
# =====================================

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
            df[col],
            errors="coerce"
        )

    df = df.dropna(subset=["Country"])

    return df, mbti_columns


try:

    df, mbti_columns = load_data()

except FileNotFoundError:

    st.error(
        f"CSV 파일을 찾을 수 없습니다.\n\n"
        f"현재 검색 경로: {CSV_PATH}"
    )

    st.stop()

except Exception as e:

    st.error(f"데이터 오류: {e}")

    st.stop()


# =====================================
# 4. 그래프 색상 함수
# =====================================

def make_colors(count):

    if count <= 0:
        return []

    colors = ["#E53935"]

    start = (29, 78, 216)
    end = (234, 248, 254)

    for i in range(count - 1):

        ratio = (
            i / (count - 2)
            if count > 2
            else 0
        )

        r = int(
            start[0] +
            (end[0] - start[0]) * ratio
        )

        g = int(
            start[1] +
            (end[1] - start[1]) * ratio
        )

        b = int(
            start[2] +
            (end[2] - start[2]) * ratio
        )

        colors.append(
            f"#{r:02X}{g:02X}{b:02X}"
        )

    return colors


# =====================================
# 5. 비율 계산 함수
# =====================================

def convert_percentage(series):

    series = pd.to_numeric(
        series,
        errors="coerce"
    )

    if series.max() <= 1:
        return series * 100

    return series


# =====================================
# 6. 탭 생성
# =====================================

tab1, tab2 = st.tabs([
    "🌍 국가별 MBTI 분석",
    "🧠 MBTI별 국가 순위"
])


# =====================================
# 7. 국가별 MBTI 분석
# =====================================

with tab1:

    st.header("국가별 MBTI 분석")

    countries = sorted(
        df["Country"].astype(str).unique()
    )

    selected_country = st.selectbox(
        "국가를 선택하세요",
        countries,
        key="country_select"
    )

    country_data = df[
        df["Country"].astype(str)
        == selected_country
    ].iloc[0]

    chart_df = pd.DataFrame({

        "MBTI": mbti_columns,

        "비율": [
            country_data[col]
            for col in mbti_columns
        ]

    })

    chart_df = chart_df.dropna()

    chart_df["비율(%)"] = convert_percentage(
        chart_df["비율"]
    )

    chart_df = chart_df.sort_values(
        "비율(%)",
        ascending=False
    ).reset_index(drop=True)

    st.subheader(
        f"📊 {selected_country}의 MBTI 비율"
    )

    if not chart_df.empty:

        col1, col2 = st.columns(2)

        col1.metric(
            "가장 높은 MBTI",
            chart_df.iloc[0]["MBTI"]
        )

        col2.metric(
            "1위 비율",
            f'{chart_df.iloc[0]["비율(%)"]:.2f}%'
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(

                x=chart_df["MBTI"],

                y=chart_df["비율(%)"],

                marker=dict(
                    color=make_colors(
                        len(chart_df)
                    )
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
        )

        fig.update_layout(

            title=f"{selected_country} MBTI 유형별 비율",

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

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        with st.expander("상세 데이터 보기"):

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


# =====================================
# 8. MBTI별 국가 순위
# =====================================

with tab2:

    st.header("MBTI별 국가 순위")

    selected_mbti = st.selectbox(
        "MBTI 유형을 선택하세요",
        mbti_columns,
        key="mbti_select"
    )

    ranking_df = df[
        ["Country", selected_mbti]
    ].copy()

    ranking_df = ranking_df.dropna(
        subset=[selected_mbti]
    )

    ranking_df["비율(%)"] = convert_percentage(
        ranking_df[selected_mbti]
    )

    ranking_df = ranking_df.sort_values(
        "비율(%)",
        ascending=False
    ).reset_index(drop=True)

    ranking_df.insert(
        0,
        "순위",
        range(1, len(ranking_df) + 1)
    )

    st.subheader(
        f"🏆 {selected_mbti} 국가별 순위"
    )

    if not ranking_df.empty:

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

        show_count = st.slider(
            "그래프에 표시할 국가 수",
            min_value=1,
            max_value=len(ranking_df),
            value=min(20, len(ranking_df))
        )

        top_countries = ranking_df.head(
            show_count
        ).copy()

        fig = go.Figure()

        fig.add_trace(
            go.Bar(

                x=top_countries["비율(%)"],

                y=top_countries["Country"],

                orientation="h",

                marker=dict(
                    color=make_colors(
                        len(top_countries)
                    )
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

            height=max(
                550,
                show_count * 28
            ),

            plot_bgcolor="white",

            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("📋 전체 국가 순위표")

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


# =====================================
# 9. 안내 문구
# =====================================

st.divider()

st.caption(
    "본 프로그램은 제공된 CSV 데이터를 기반으로 분석합니다. "
    "실제 인구 전체의 MBTI 분포를 의미하지는 않습니다."
)
