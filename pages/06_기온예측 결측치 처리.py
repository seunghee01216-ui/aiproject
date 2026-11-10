
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="기온 분석 및 미래 예측",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 과거 기온 분석 및 미래 예측")
st.markdown("""
1907년부터 2025년까지의 기온 데이터를 분석하고,
결측치 처리 방법과 인공지능 알고리즘에 따른
미래 평균기온 예측 결과를 비교합니다.
""")

st.divider()


# ==========================================
# 2. CSV 파일 경로 설정
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "temperature(1).csv"


# ==========================================
# 3. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}"
        )

    try:
        data = pd.read_csv(
            CSV_PATH,
            encoding="cp949"
        )
    except UnicodeDecodeError:
        data = pd.read_csv(
            CSV_PATH,
            encoding="utf-8-sig"
        )

    required_columns = ["일시", "평균기온(℃)"]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(
                f"CSV 파일에 '{column}' 열이 없습니다."
            )

    month_dict = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12
    }

    def convert_date(value):

        month_name, year_text = str(value).strip().split("-")

        month = month_dict[month_name]
        short_year = int(year_text)

        if short_year >= 7:
            year = 1900 + short_year
        else:
            year = 2000 + short_year

        return pd.Timestamp(
            year=year,
            month=month,
            day=1
        )

    data["날짜"] = data["일시"].apply(convert_date)

    data["연도"] = data["날짜"].dt.year
    data["월"] = data["날짜"].dt.month

    data["평균기온(℃)"] = pd.to_numeric(
        data["평균기온(℃)"],
        errors="coerce"
    )

    # 분석 범위 설정
    data = data[
        (data["연도"] >= 1907) &
        (data["연도"] <= 2025)
    ].copy()

    data = data.sort_values("날짜")

    return data.reset_index(drop=True)


try:
    df = load_data()

except Exception as e:
    st.error(f"데이터 불러오기 오류: {e}")
    st.stop()


# ==========================================
# 4. 데이터 기본 정보
# ==========================================

st.header("📊 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

col1.metric("분석 시작", "1907년")
col2.metric("분석 종료", "2025년")

missing_count = df["평균기온(℃)"].isna().sum()

col3.metric("결측치 개수", f"{missing_count}개")

st.caption(f"CSV 파일 위치: {CSV_PATH.name}")

with st.expander("원본 데이터 확인하기"):

    st.dataframe(
        df[["일시", "평균기온(℃)"]],
        use_container_width=True
    )


# ==========================================
# 5. 월 선택
# ==========================================

st.divider()

st.header("📅 분석할 월 선택")

selected_month = st.selectbox(
    "분석할 월을 선택하세요.",
    list(range(1, 13)),
    format_func=lambda x: f"{x}월"
)


# ==========================================
# 6. 과거 평균기온 그래프
# ==========================================

monthly_data = df[
    df["월"] == selected_month
].copy()

st.subheader(
    f"📈 {selected_month}월 과거 평균기온"
)

fig1, ax1 = plt.subplots(figsize=(12, 5))

ax1.plot(
    monthly_data["연도"],
    monthly_data["평균기온(℃)"],
    color="royalblue",
    linewidth=1.5,
    marker="o",
    markersize=2,
    label="실제 평균기온"
)

ax1.set_title(
    f"{selected_month}월 평균기온 변화 (1907~2025)"
)

ax1.set_xlabel("연도")
ax1.set_ylabel("평균기온(℃)")

ax1.grid(alpha=0.3)
ax1.legend()

st.pyplot(fig1)
plt.close(fig1)


# ==========================================
# 7. 결측치 처리 방법 선택
# ==========================================

st.divider()

st.header("🧹 결측치 처리 방법 선택")

st.markdown("""
결측치란 기온 측정값이 비어 있는 데이터입니다.

처리 방법에 따라 학습에 사용되는 데이터가 달라지며,
그 결과 미래 예측값도 달라질 수 있습니다.
""")

imputation_options = [
    "삭제",
    "중앙값 대체",
    "선형보간법"
]

selected_imputations = st.multiselect(
    "비교할 결측치 처리 방법을 선택하세요.",
    imputation_options,
    default=imputation_options
)

if not selected_imputations:
    st.warning("결측치 처리 방법을 하나 이상 선택하세요.")
    st.stop()


# ==========================================
# 8. 결측치 처리 함수
# ==========================================

def process_missing_data(data, method):

    result = data.copy()

    result = result.sort_values("날짜")

    result = result.set_index("날짜")

    if method == "삭제":

        result = result.dropna(
            subset=["평균기온(℃)"]
        )

    elif method == "중앙값 대체":

        # 같은 월의 중앙값으로 대체
        monthly_median = (
            result.groupby("월")["평균기온(℃)"]
            .median()
        )

        missing_mask = result["평균기온(℃)"].isna()

        result.loc[missing_mask, "평균기온(℃)"] = (
            result.loc[missing_mask, "월"]
            .map(monthly_median)
        )

        # 월별 중앙값이 없는 경우 전체 중앙값 사용
        overall_median = data["평균기온(℃)"].median()

        result["평균기온(℃)"] = (
            result["평균기온(℃)"]
            .fillna(overall_median)
        )

    elif method == "선형보간법":

        result["평균기온(℃)"] = (
            result["평균기온(℃)"]
            .interpolate(
                method="time",
                limit_direction="both"
            )
        )

        # 남아 있는 결측치는 월별 중앙값으로 처리
        monthly_median = (
            result.groupby("월")["평균기온(℃)"]
            .transform("median")
        )

        result["평균기온(℃)"] = (
            result["평균기온(℃)"]
            .fillna(monthly_median)
        )

        result["평균기온(℃)"] = (
            result["평균기온(℃)"]
            .fillna(data["평균기온(℃)"].median())
        )

    result = result.reset_index()

    return result


# ==========================================
# 9. 결측치 처리 결과 비교
# ==========================================

st.subheader("🔍 결측치 처리 결과")

imputation_results = {}

for method in selected_imputations:

    processed = process_missing_data(
        df,
        method
    )

    imputation_results[method] = processed

    original_count = len(df)

    processed_count = len(processed)

    remaining_missing = (
        processed["평균기온(℃)"].isna().sum()
    )

    st.write(
        f"**{method}** | "
        f"처리 후 데이터: {processed_count}개 | "
        f"남은 결측치: {remaining_missing}개"
    )


# ==========================================
# 10. 결측치 처리별 그래프
# ==========================================

st.subheader("📉 결측치 처리 방법에 따른 기온 변화")

line_styles = {
    "삭제": "-",
    "중앙값 대체": "--",
    "선형보간법": ":"
}

imputation_colors = {
    "삭제": "red",
    "중앙값 대체": "blue",
    "선형보간법": "green"
}

fig2, ax2 = plt.subplots(figsize=(12, 5))

for method, processed in imputation_results.items():

    selected_data = processed[
        processed["월"] == selected_month
    ]

    ax2.plot(
        selected_data["연도"],
        selected_data["평균기온(℃)"],
        color=imputation_colors[method],
        linestyle=line_styles[method],
        linewidth=1.5,
        label=method
    )

ax2.set_title(
    f"{selected_month}월 결측치 처리 방법 비교"
)

ax2.set_xlabel("연도")
ax2.set_ylabel("평균기온(℃)")

ax2.grid(alpha=0.3)
ax2.legend()

st.pyplot(fig2)
plt.close(fig2)


# ==========================================
# 11. 알고리즘 선택
# ==========================================

st.divider()

st.header("🤖 예측 알고리즘 선택")

st.markdown("""
3가지 알고리즘 중 원하는 방법을 선택하세요.

여러 개를 동시에 선택하면 한 그래프에서 비교할 수 있습니다.
""")

algorithm_options = [
    "선형 회귀",
    "랜덤 포레스트",
    "그래디언트 부스팅"
]

selected_algorithms = st.multiselect(
    "비교할 알고리즘을 선택하세요.",
    algorithm_options,
    default=algorithm_options
)

if not selected_algorithms:
    st.warning("예측 알고리즘을 하나 이상 선택하세요.")
    st.stop()


# ==========================================
# 12. 알고리즘 설명
# ==========================================

st.subheader("🧠 알고리즘 작동 원리")

with st.expander("🔴 선형 회귀 (Linear Regression)"):

    st.markdown("""
    **작동 원리**

    - 과거 기온의 전체적인 상승 또는 하락 추세를 분석합니다.
    - 시간과 기온의 관계를 직선 형태로 학습합니다.
    - 계절에 따른 변화도 함께 반영합니다.

    **그래프 특징**

    - 비교적 부드러운 직선 형태의 추세가 나타납니다.

    **장점:** 빠르고 이해하기 쉽습니다.

    **단점:** 복잡한 기온 변화 패턴을 표현하기 어렵습니다.
    """)

with st.expander("🔵 랜덤 포레스트 (Random Forest)"):

    st.markdown("""
    **작동 원리**

    - 여러 개의 의사결정나무를 생성합니다.
    - 각각의 나무가 예측한 결과를 평균합니다.
    - 여러 나무의 결과를 모아 최종 예측값을 만듭니다.

    **그래프 특징**

    - 과거 데이터의 다양한 패턴을 학습할 수 있습니다.
    - 미래로 갈수록 학습 범위의 예측값에 가까워질 수 있습니다.

    **장점:** 복잡한 데이터 관계를 학습할 수 있습니다.

    **단점:** 장기적인 미래 추세를 외삽하는 데 약합니다.
    """)

with st.expander("🟢 그래디언트 부스팅 (Gradient Boosting)"):

    st.markdown("""
    **작동 원리**

    - 처음에는 간단한 예측 모델을 만듭니다.
    - 이전 모델이 틀린 부분을 다음 모델이 보완합니다.
    - 여러 번 반복하여 예측 성능을 높입니다.

    **그래프 특징**

    - 데이터의 복잡한 패턴을 학습할 수 있습니다.
    - 랜덤 포레스트와 다른 예측 곡선이 나타날 수 있습니다.

    **장점:** 여러 단계의 보정을 통해 패턴을 학습합니다.

    **단점:** 설정에 따라 과적합이 발생할 수 있습니다.
    """)


# ==========================================
# 13. 미래 예측 연도 선택
# ==========================================

st.divider()

st.header("🔮 미래 평균기온 예측")

target_year = st.number_input(
    "예측하고 싶은 미래 연도를 입력하세요.",
    min_value=2026,
    max_value=2100,
    value=2035,
    step=1
)


# ==========================================
# 14. 학습 특징 생성
# ==========================================

def make_features(years, months):

    years = np.asarray(years)
    months = np.asarray(months)

    return pd.DataFrame({

        "year_trend": years - 1907,

        "month_sin": np.sin(
            2 * np.pi * months / 12
        ),

        "month_cos": np.cos(
            2 * np.pi * months / 12
        )
    })


# ==========================================
# 15. 알고리즘 생성
# ==========================================

def create_model(name):

    if name == "선형 회귀":

        return LinearRegression()

    elif name == "랜덤 포레스트":

        return RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42
        )

    elif name == "그래디언트 부스팅":

        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2,
            loss="huber",
            random_state=42
        )


# ==========================================
# 16. 미래 예측 데이터 생성
# ==========================================

future_years = np.arange(
    2026,
    int(target_year) + 1
)

future_months = np.full(
    len(future_years),
    selected_month
)

future_X = make_features(
    future_years,
    future_months
)


# ==========================================
# 17. 결측치 + 알고리즘별 학습
# ==========================================

predictions = {}

model_colors = {
    "선형 회귀": "red",
    "랜덤 포레스트": "blue",
    "그래디언트 부스팅": "green"
}

for method in selected_imputations:

    processed = imputation_results[method]

    X_train = make_features(
        processed["연도"].values,
        processed["월"].values
    )

    y_train = processed["평균기온(℃)"].values

    if len(y_train) < 10:
        st.error(
            f"{method} 방법의 학습 데이터가 부족합니다."
        )
        st.stop()

    for algorithm in selected_algorithms:

        model = create_model(algorithm)

        model.fit(
            X_train,
            y_train
        )

        predicted = model.predict(future_X)

        predictions[(method, algorithm)] = predicted


# ==========================================
# 18. 미래 예측 그래프
# ==========================================

st.divider()

st.header("📈 미래 기온 예측 결과 비교")

st.markdown("""
**그래프 읽는 방법**

- 검정색 실선: 과거 실제 기온
- 빨간색: 선형 회귀
- 파란색: 랜덤 포레스트
- 초록색: 그래디언트 부스팅
- 실선: 결측치 삭제
- 파선: 중앙값 대체
- 점선: 선형보간법
""")

fig3, ax3 = plt.subplots(figsize=(14, 7))

# 과거 실제 기온
ax3.plot(
    monthly_data["연도"],
    monthly_data["평균기온(℃)"],
    color="black",
    linewidth=2,
    label="실제 기온",
    zorder=5
)

# 미래 예측 결과
for (method, algorithm), predicted in predictions.items():

    label = f"{algorithm} / {method}"

    ax3.plot(
        future_years,
        predicted,
        color=model_colors[algorithm],
        linestyle=line_styles[method],
        linewidth=2,
        label=label
    )

ax3.axvline(
    x=2025,
    color="gray",
    linestyle="-.",
    label="예측 시작"
)

ax3.set_title(
    f"{selected_month}월 미래 평균기온 예측"
)

ax3.set_xlabel("연도")
ax3.set_ylabel("평균기온(℃)")

ax3.grid(alpha=0.3)
ax3.legend(fontsize=8)

st.pyplot(fig3)
plt.close(fig3)


# ==========================================
# 19. 예측 결과 표
# ==========================================

st.subheader("📋 예측 결과 비교표")

result_rows = []

for (method, algorithm), predicted in predictions.items():

    result_rows.append({

        "결측치 처리 방법": method,

        "예측 알고리즘": algorithm,

        "예측 연도": int(target_year),

        "예측 월": f"{selected_month}월",

        "예측 평균기온(℃)": round(
            float(predicted[-1]),
            2
        )
    })

result_df = pd.DataFrame(result_rows)

st.dataframe(
    result_df,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 20. 예측값 차이 분석
# ==========================================

st.divider()

st.header("📊 예측값 차이 분석")

if len(result_df) > 1:

    max_value = result_df["예측 평균기온(℃)"].max()

    min_value = result_df["예측 평균기온(℃)"].min()

    difference = max_value - min_value

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "최고 예측값",
        f"{max_value:.2f} ℃"
    )

    col2.metric(
        "최저 예측값",
        f"{min_value:.2f} ℃"
    )

    col3.metric(
        "예측값 차이",
        f"{difference:.2f} ℃"
    )

    # 막대그래프
    fig4, ax4 = plt.subplots(figsize=(12, 5))

    labels = [
        f"{row['예측 알고리즘']}\n{row['결측치 처리 방법']}"
        for _, row in result_df.iterrows()
    ]

    colors = [
        model_colors[name]
        for name in result_df["예측 알고리즘"]
    ]

    bars = ax4.bar(
        labels,
        result_df["예측 평균기온(℃)"],
        color=colors,
        alpha=0.75
    )

    ax4.set_title(
        f"{target_year}년 {selected_month}월 예측값 비교"
    )

    ax4.set_ylabel("평균기온(℃)")
    ax4.grid(axis="y", alpha=0.3)

    for bar in bars:

        height = bar.get_height()

        ax4.annotate(
            f"{height:.2f}℃",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center"
        )

    st.pyplot(fig4)
    plt.close(fig4)

else:

    st.info("예측값 차이를 비교하려면 두 개 이상의 조합을 선택하세요.")


# ==========================================
# 21. 분석 결과 해석
# ==========================================

st.divider()

st.header("💡 분석 결과 이해하기")

st.subheader("결측치 처리 방법의 차이")

st.markdown("""
| 방법 | 작동 원리 | 특징 |
|---|---|---|
| 삭제 | 결측 행 제거 | 데이터 감소 |
| 중앙값 대체 | 같은 월의 중앙값 사용 | 계절 특성 유지 |
| 선형보간법 | 앞뒤 데이터로 계산 | 시간적 흐름 반영 |
""")

st.subheader("예측 알고리즘의 차이")

st.markdown("""
| 알고리즘 | 작동 원리 | 특징 |
|---|---|---|
| 선형 회귀 | 직선 추세 학습 | 장기 추세 확인 |
| 랜덤 포레스트 | 여러 나무의 평균 | 복잡한 패턴 학습 |
| 그래디언트 부스팅 | 오차 반복 보정 | 단계적인 학습 |
""")

st.warning("""
미래 기온 예측은 과거 데이터의 패턴을 이용한 통계적 결과입니다.
실제 미래 기온과 다를 수 있습니다.

특히 랜덤 포레스트와 그래디언트 부스팅은
학습 데이터 범위를 벗어난 장기적인 미래 추세를
정확하게 예측하는 데 한계가 있습니다.
""")

st.success("🌡️ 기온 분석 및 미래 예측이 완료되었습니다!")
