
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# --------------------------------
# 1. 페이지 설정
# --------------------------------

st.set_page_config(
    page_title="미래 기온 예측 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 과거 기온 분석 및 미래 기온 예측")
st.write(
    "1907년부터 2025년까지의 기온 데이터를 분석하고 "
    "3가지 인공지능 알고리즘으로 미래 기온을 예측합니다."
)

st.divider()


# --------------------------------
# 2. CSV 파일 불러오기
# --------------------------------

@st.cache_data
def load_data():

    file_path = (
        Path(__file__).resolve().parent.parent
        / "temperature(1).csv"
    )

    if not file_path.exists():
        st.error("CSV 파일을 찾을 수 없습니다.")
        st.stop()

    try:
        data = pd.read_csv(file_path, encoding="cp949")
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding="utf-8-sig")

    # 날짜 변환
    month_dict = {
        "Jan": 1, "Feb": 2, "Mar": 3,
        "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9,
        "Oct": 10, "Nov": 11, "Dec": 12
    }

    def convert_date(value):
        month_name, year_text = str(value).split("-")

        month = month_dict[month_name]
        short_year = int(year_text)

        # 1907~1999년과 2000~2026년 구분
        if short_year >= 7:
            year = 1900 + short_year
        else:
            year = 2000 + short_year

        return pd.Timestamp(year=year, month=month, day=1)

    data["날짜"] = data["일시"].apply(convert_date)

    data["연도"] = data["날짜"].dt.year
    data["월"] = data["날짜"].dt.month

    data["평균기온(℃)"] = pd.to_numeric(
        data["평균기온(℃)"],
        errors="coerce"
    )

    # 분석 범위: 1907~2025년
    data = data[
        (data["연도"] >= 1907) &
        (data["연도"] <= 2025)
    ].copy()

    data = data.dropna(subset=["평균기온(℃)"])

    return data


df = load_data()


# --------------------------------
# 3. 월 선택
# --------------------------------

st.subheader("📅 분석할 월 선택")

selected_month = st.selectbox(
    "분석하고 싶은 월을 선택하세요.",
    list(range(1, 13)),
    format_func=lambda x: f"{x}월"
)


# --------------------------------
# 4. 선택한 월의 과거 기온
# --------------------------------

monthly_data = df[
    df["월"] == selected_month
].sort_values("연도")

st.subheader(
    f"📈 {selected_month}월의 1907~2025년 평균기온"
)

fig1, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(
    monthly_data["연도"],
    monthly_data["평균기온(℃)"],
    color="royalblue",
    linewidth=1.5,
    label="실제 평균기온"
)

ax1.set_title(
    f"{selected_month}월 Historical Temperature"
)

ax1.set_xlabel("Year")
ax1.set_ylabel("Temperature (℃)")
ax1.grid(alpha=0.3)
ax1.legend()

st.pyplot(fig1)
plt.close(fig1)


# --------------------------------
# 5. 머신러닝 학습 데이터 만들기
# --------------------------------

# 시간과 계절을 나타내는 특징 생성

def make_features(years, months):

    years = np.asarray(years)
    months = np.asarray(months)

    return pd.DataFrame({
        "year_trend": years - 1907,
        "month_sin": np.sin(2 * np.pi * months / 12),
        "month_cos": np.cos(2 * np.pi * months / 12)
    })


X = make_features(
    df["연도"].values,
    df["월"].values
)

y = df["평균기온(℃)"].values


# --------------------------------
# 6. 알고리즘 3개 생성
# --------------------------------

models = {

    "선형 회귀": LinearRegression(),

    "랜덤 포레스트": RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42
    ),

    "그래디언트 부스팅": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=2,
        loss="huber",
        random_state=42
    )
}


# --------------------------------
# 7. 모델 학습
# --------------------------------

for model in models.values():
    model.fit(X, y)


# --------------------------------
# 8. 미래 연도 입력
# --------------------------------

st.divider()

st.subheader("🔮 미래 기온 예측")

target_year = st.number_input(
    "예측하고 싶은 미래 연도를 입력하세요.",
    min_value=2026,
    max_value=2100,
    value=2035,
    step=1
)

st.caption(
    "선택한 월을 기준으로 해당 연도까지의 "
    "미래 평균기온을 예측합니다."
)


# --------------------------------
# 9. 미래 예측 데이터 생성
# --------------------------------

future_years = np.arange(2026, target_year + 1)

future_months = np.full(
    len(future_years),
    selected_month
)

future_X = make_features(
    future_years,
    future_months
)

predictions = {}

for name, model in models.items():

    predictions[name] = model.predict(future_X)


# --------------------------------
# 10. 실제 기온과 미래 예측 비교
# --------------------------------

st.subheader(
    f"🌍 {selected_month}월 미래 기온 예측 비교"
)

fig2, ax2 = plt.subplots(figsize=(13, 6))

# 과거 실제 기온
ax2.plot(
    monthly_data["연도"],
    monthly_data["평균기온(℃)"],
    color="black",
    linewidth=1.8,
    label="실제 기온"
)

# 예측 알고리즘 3개
colors = {
    "선형 회귀": "red",
    "랜덤 포레스트": "blue",
    "그래디언트 부스팅": "green"
}

for name, prediction in predictions.items():

    ax2.plot(
        future_years,
        prediction,
        color=colors[name],
        linewidth=2,
        linestyle="--",
        label=name
    )

# 예측 시작 지점 표시
ax2.axvline(
    x=2025,
    color="gray",
    linestyle=":",
    label="예측 시작"
)

ax2.set_title(
    f"{selected_month}월 Temperature Prediction"
)

ax2.set_xlabel("Year")
ax2.set_ylabel("Average Temperature (℃)")

ax2.grid(alpha=0.3)
ax2.legend()

st.pyplot(fig2)
plt.close(fig2)


# --------------------------------
# 11. 최종 예측 결과
# --------------------------------

st.subheader("📊 예측 결과 비교")

result_data = pd.DataFrame({
    "알고리즘": list(predictions.keys()),
    "예측 평균기온(℃)": [
        round(float(pred[-1]), 2)
        for pred in predictions.values()
    ]
})

st.dataframe(
    result_data,
    use_container_width=True,
    hide_index=True
)

st.caption(
    f"{target_year}년 {selected_month}월의 예측 결과입니다."
)


# --------------------------------
# 12. 알고리즘별 설명
# --------------------------------

st.divider()

st.header("🧠 예측 알고리즘 알아보기")

st.write(
    "세 가지 알고리즘은 과거 기온 데이터를 학습하지만 "
    "미래를 예측하는 방법은 서로 다릅니다."
)

# 선형 회귀
with st.expander("🔴 1. 선형 회귀 (Linear Regression)"):

    st.write("""
    **작동 원리**

    - 과거 데이터에서 전체적인 상승 또는 하락 추세를 찾습니다.
    - 시간에 따른 기온 변화를 직선 형태로 표현합니다.
    - 계절 효과도 함께 반영합니다.

    **그래프에서 보는 특징**

    - 비교적 부드러운 직선 형태의 변화가 나타납니다.
    - 장기적인 온도 변화 추세를 확인하기 좋습니다.

    **장점**
    - 계산이 빠르고 이해하기 쉽습니다.

    **단점**
    - 복잡하고 급격한 기온 변화는 충분히 표현하지 못할 수 있습니다.
    """)


# 랜덤 포레스트
with st.expander("🔵 2. 랜덤 포레스트 (Random Forest)"):

    st.write("""
    **작동 원리**

    - 여러 개의 의사결정나무를 만듭니다.
    - 각 나무가 예측한 결과를 평균 내어 최종 결과를 만듭니다.
    - 다양한 데이터 패턴을 학습할 수 있습니다.

    **그래프에서 보는 특징**

    - 과거 데이터의 복잡한 변화 패턴을 반영할 수 있습니다.
    - 미래로 갈수록 학습 데이터의 범위를 벗어난 예측은
      일정한 값에 가까워질 수 있습니다.

    **장점**
    - 복잡한 데이터 관계를 학습할 수 있습니다.

    **단점**
    - 학습 범위를 벗어난 미래 추세를 외삽하는 데 약합니다.
    """)


# 그래디언트 부스팅
with st.expander("🟢 3. 그래디언트 부스팅 (Gradient Boosting)"):

    st.write("""
    **작동 원리**

    - 처음에는 간단한 예측 모델을 만듭니다.
    - 이전 모델이 틀린 부분을 다음 모델이 보완합니다.
    - 이 과정을 반복하여 예측 성능을 높입니다.

    **그래프에서 보는 특징**

    - 데이터의 비선형적인 패턴을 학습할 수 있습니다.
    - 랜덤 포레스트와는 다른 방식으로 예측 곡선이 만들어집니다.
    - 미래 데이터가 학습 범위를 벗어나면 추세가 제한될 수 있습니다.

    **장점**
    - 여러 번의 보정을 통해 복잡한 패턴을 학습합니다.

    **단점**
    - 설정에 따라 과적합이 발생할 수 있습니다.
    """)


# --------------------------------
# 13. 알고리즘 비교
# --------------------------------

st.subheader("🔍 세 알고리즘의 차이")

comparison = pd.DataFrame({
    "비교 항목": [
        "예측 방법",
        "주요 특징",
        "장점",
        "한계"
    ],

    "선형 회귀": [
        "직선 추세 학습",
        "전체적인 변화",
        "간단하고 빠름",
        "복잡한 패턴에 약함"
    ],

    "랜덤 포레스트": [
        "여러 나무의 평균",
        "다양한 패턴 학습",
        "비선형 관계 학습",
        "미래 외삽에 약함"
    ],

    "그래디언트 부스팅": [
        "오차를 반복 보정",
        "단계적인 학습",
        "복잡한 패턴 학습",
        "과적합 가능성"
    ]
})

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)


# --------------------------------
# 14. 데이터 안내
# --------------------------------

st.divider()

st.subheader("📌 분석 안내")

st.write("""
- 분석 데이터: 업로드된 월별 평균기온 데이터
- 학습 기간: 1907년~2025년
- 예측 대상: 사용자가 선택한 월의 평균기온
- 예측 알고리즘: 선형 회귀, 랜덤 포레스트, 그래디언트 부스팅

**주의사항**

미래 기온은 과거 데이터의 패턴을 이용한 통계적 예측입니다.
실제 미래 기온과 다를 수 있으며,
기후변화, 대기 순환, 해양 현상 등의 모든 요인을
반영하는 기후 예측 모델은 아닙니다.
""")

st.success("기온 분석 및 미래 예측이 완료되었습니다! 🌡️")
