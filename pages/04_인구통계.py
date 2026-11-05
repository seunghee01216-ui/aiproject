
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import math

# =====================================
# 1. 페이지 설정
# =====================================
st.set_page_config(
    page_title="경기도 연령별 인구 통계",
    page_icon="📊",
    layout="wide"
)

st.title("📊 경기도 연령별 인구 통계")
st.write("지역, 성별, 연도를 선택하여 연령대별 인구를 확인해 보세요!")

# =====================================
# 2. 파일 경로 설정
# =====================================
# 현재 파일은 pages 폴더 안에 있음
# 상위 폴더에서 population.csv 찾기

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "population.csv"

# =====================================
# 3. CSV 데이터 불러오기
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv(
        CSV_PATH,
        encoding="cp949",
        thousands=","
    )

    df.columns = df.columns.str.strip()
    df["행정구역"] = df["행정구역"].astype(str)

    return df


try:
    df = load_data()

except FileNotFoundError:
    st.error("population.csv 파일을 찾을 수 없습니다.")
    st.error(f"확인한 경로: {CSV_PATH}")
    st.stop()

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# =====================================
# 4. 데이터 설정
# =====================================
years = list(range(2020, 2026))

genders = {
    "전체": "계",
    "남성": "남",
    "여성": "여"
}

ages = [
    "0~9세",
    "10~19세",
    "20~29세",
    "30~39세",
    "40~49세",
    "50~59세",
    "60~69세",
    "70~79세",
    "80~89세",
    "90~99세",
    "100세 이상"
]

# =====================================
# 5. 행정구역 선택
# =====================================
st.subheader("🔎 검색 조건")

regions = df["행정구역"].tolist()

selected_region = st.selectbox(
    "행정구역을 선택하세요",
    regions
)

# =====================================
# 6. 성별 및 연도 선택
# =====================================
col1, col2 = st.columns(2)

with col1:
    selected_gender = st.selectbox(
        "성별을 선택하세요",
        list(genders.keys())
    )

with col2:
    selected_year = st.selectbox(
        "연도를 선택하세요",
        years,
        index=len(years) - 1
    )

gender_code = genders[selected_gender]

# =====================================
# 7. 선택한 데이터 가져오기
# =====================================
selected_row = df[
    df["행정구역"] == selected_region
].iloc[0]

population = []

for age in ages:

    column_name = (
        f"{selected_year}년_{gender_code}_{age}"
    )

    value = pd.to_numeric(
        selected_row[column_name],
        errors="coerce"
    )

    population.append(
        0 if pd.isna(value) else int(value)
    )

# =====================================
# 8. 데이터프레임 생성
# =====================================
chart_df = pd.DataFrame({
    "연령대": ages,
    "인구수": population
})

total_population = sum(population)

# =====================================
# 9. 인구 통계 요약
# =====================================
st.divider()
st.subheader("📌 인구 통계 요약")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "선택 지역",
        selected_region.split(" (")[0]
    )

with col2:
    st.metric(
        "선택 연도",
        f"{selected_year}년"
    )

with col3:
    st.metric(
        "연령별 인구 합계",
        f"{total_population:,}명"
    )

# =====================================
# 10. 꺾은선 그래프
# =====================================
st.divider()
st.subheader("📈 연령대별 인구 꺾은선 그래프")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_df["연령대"],
        y=chart_df["인구수"],
        mode="lines+markers",
        name="인구수",

        line=dict(
            color="#2563EB",
            width=3
        ),

        marker=dict(
            color="#2563EB",
            size=8
        ),

        hovertemplate=(
            "연령대: %{x}<br>"
            "인구수: %{y:,.0f}명"
            "<extra></extra>"
        )
    )
)

# =====================================
# 11. 세로축 설정
# 1,000만 명 단위
# =====================================
max_population = max(population) if population else 0

y_max = max(
    10_000_000,
    math.ceil(max_population / 10_000_000)
    * 10_000_000
)

fig.update_layout(

    xaxis=dict(
        title="연령대",
        type="category",
        tickangle=0,
        showgrid=True,
        dtick=1,
        gridcolor="#D9D9D9"
    ),

    yaxis=dict(
        title="인구수 (명)",
        range=[0, y_max],
        tick0=0,
        dtick=10_000_000,
        tickformat=",",
        showgrid=True,
        gridcolor="#D9D9D9"
    ),

    height=550,

    hovermode="x unified",

    plot_bgcolor="white",
    paper_bgcolor="white",

    font=dict(
        family="Malgun Gothic, Apple SD Gothic Neo, sans-serif",
        size=13,
        color="#333333"
    ),

    margin=dict(
        l=80,
        r=30,
        t=30,
        b=70
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# 12. 연령별 상세 인구
# =====================================
st.divider()
st.subheader("📋 연령대별 상세 인구")

display_df = chart_df.copy()

display_df["인구수"] = display_df["인구수"].apply(
    lambda x: f"{x:,}명"
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# =====================================
# 13. 데이터 안내
# =====================================
st.divider()

st.caption(
    "자료: population.csv | "
    "연령대별 인구는 선택한 지역, 연도, 성별 기준입니다."
)
