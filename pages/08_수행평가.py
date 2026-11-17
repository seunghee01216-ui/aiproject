
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="AI Chocolate Recommend",
    page_icon="🍫",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #FFF9F2;
}
h1, h2, h3 {
    color: #683D27;
}
.stButton>button {
    background-color: #79452D;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🍫 AI Chocolate Recommend")
st.subheader("나의 취향을 분석하는 인공지능 초콜릿 추천소")

st.write(
    "원하는 초콜릿 취향을 선택하면 "
    "인공지능이 가장 잘 어울리는 초콜릿을 추천해 줍니다!"
)

st.markdown("---")


# =====================================
# 2. CSV 데이터 불러오기
# =====================================

@st.cache_data
def load_data():

    # 현재 파일은 pages 폴더에 있음
    current_folder = Path(__file__).resolve().parent

    # CSV 파일은 pages의 상위 폴더에 있음
    parent_folder = current_folder.parent

    possible_files = [
        parent_folder / "flavors_of_cacao.csv",
        parent_folder / "flavors_of_cacao.csv"
    ]

    # 파일 이름이 조금 달라도 자동 탐색
    possible_files += list(
        parent_folder.glob("flavors_of_cacao*.csv")
    )

    file_path = next(
        (p for p in possible_files if p.exists()),
        None
    )

    if file_path is None:
        raise FileNotFoundError(
            "상위 폴더에서 flavors_of_cacao CSV 파일을 찾을 수 없습니다."
        )

    data = pd.read_csv(file_path)

    # 열 이름의 줄바꿈과 특수 공백 제거
    def normalize_column(name):
        return re.sub(
            r"[^a-z0-9]",
            "",
            str(name).lower()
        )

    # 실제 CSV 열 이름과 사용할 이름 연결
    column_mapping = {
        "companymakerifknown": "Company",
        "specificbeanoriginorbarname": "Bean_Origin",
        "ref": "REF",
        "reviewdate": "Review_Date",
        "cocoapercent": "Cocoa_Percent",
        "companylocation": "Company_Location",
        "rating": "Rating",
        "beantype": "Bean_Type",
        "broadbeanorigin": "Broad_Origin"
    }

    new_columns = {}

    for col in data.columns:
        key = normalize_column(col)

        if key in column_mapping:
            new_columns[col] = column_mapping[key]

    data = data.rename(columns=new_columns)

    # 필수 열이 누락된 경우에도 오류 방지
    required_columns = [
        "Company",
        "Bean_Origin",
        "Cocoa_Percent",
        "Rating",
        "Bean_Type",
        "Broad_Origin",
        "Review_Date"
    ]

    for col in required_columns:
        if col not in data.columns:
            data[col] = np.nan

    # 카카오 함량 숫자 변환
    data["Cocoa_Percent"] = (
        data["Cocoa_Percent"]
        .astype(str)
        .str.replace("%", "", regex=False)
    )

    data["Cocoa_Percent"] = pd.to_numeric(
        data["Cocoa_Percent"],
        errors="coerce"
    )

    data["Rating"] = pd.to_numeric(
        data["Rating"],
        errors="coerce"
    )

    data["Review_Date"] = pd.to_numeric(
        data["Review_Date"],
        errors="coerce"
    )

    # 문자 데이터 정리
    for col in [
        "Company",
        "Bean_Origin",
        "Bean_Type",
        "Broad_Origin"
    ]:
        data[col] = (
            data[col]
            .astype("string")
            .str.strip()
        )

        data[col] = data[col].replace(
            ["", "nan", "None", "<NA>"],
            pd.NA
        )

        data[col] = data[col].fillna("정보 없음")

    # 학습에 필요한 데이터가 있는 행만 사용
    data = data.dropna(
        subset=["Cocoa_Percent", "Rating"]
    )

    data = data[
        (data["Cocoa_Percent"] >= 0) &
        (data["Cocoa_Percent"] <= 100) &
        (data["Rating"] >= 0) &
        (data["Rating"] <= 5)
    ].copy()

    return data


try:
    df = load_data()

except Exception as e:
    st.error(f"데이터 불러오기 오류: {e}")
    st.stop()


# =====================================
# 3. 인공지능 모델 학습
# =====================================

categorical_cols = [
    "Company",
    "Bean_Origin",
    "Bean_Type",
    "Broad_Origin"
]

numeric_cols = [
    "Cocoa_Percent",
    "Review_Date"
]

feature_cols = categorical_cols + numeric_cols


@st.cache_resource
def train_ai_model(data):

    X = data[feature_cols]
    y = data["Rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 문자 데이터 전처리
    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    # 숫자 데이터 전처리
    numeric_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "cat",
            categorical_transformer,
            categorical_cols
        ),
        (
            "num",
            numeric_transformer,
            numeric_cols
        )
    ])

    # 랜덤 포레스트 회귀 모델
    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return model, mae, r2


if len(df) < 10:
    st.error("AI 학습에 필요한 데이터가 부족합니다.")
    st.stop()

model, mae, r2 = train_ai_model(df)


# =====================================
# 4. 사이드바 메뉴
# =====================================

menu = st.sidebar.radio(
    "🍫 메뉴",
    [
        "🏠 홈",
        "🍬 AI 초콜릿 추천",
        "📊 초콜릿 데이터 분석",
        "🏆 인기 초콜릿 순위",
        "🤖 AI 학습 결과"
    ]
)


# =====================================
# 5. 홈 화면
# =====================================

if menu == "🏠 홈":

    st.header("🍫 AI Chocolate Recommend")

    st.image(
        "https://images.unsplash.com/photo-1548907040-4d42fdd6f7b1?w=1200",
        use_container_width=True
    )

    st.write("""
    이 사이트는 초콜릿 데이터를 인공지능으로 분석하여
    사용자의 취향에 맞는 초콜릿을 추천하는 서비스입니다.

    카카오 함량과 원산지 등의 정보를 선택하면
    AI가 예상 평점을 계산하고 어울리는 초콜릿을 찾아줍니다.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "분석 데이터",
        f"{len(df):,}개"
    )

    col2.metric(
        "평균 카카오 함량",
        f"{df['Cocoa_Percent'].mean():.1f}%"
    )

    col3.metric(
        "평균 평점",
        f"{df['Rating'].mean():.2f}점"
    )

    st.info(
        "왼쪽 메뉴에서 AI 초콜릿 추천을 선택해 보세요!"
    )


# =====================================
# 6. AI 초콜릿 추천
# =====================================

elif menu == "🍬 AI 초콜릿 추천":

    st.header("🍬 나만의 초콜릿 추천")

    st.write(
        "원하는 초콜릿 취향을 선택해 주세요. "
        "AI가 취향에 맞는 초콜릿 TOP 5를 추천합니다."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        cocoa_preference = st.slider(
            "🍫 원하는 카카오 함량",
            min_value=40,
            max_value=100,
            value=70,
            step=1
        )

        st.caption(
            f"선택한 카카오 함량: {cocoa_preference}%"
        )

    with col2:

        origin_list = sorted(
            df["Broad_Origin"].unique().tolist()
        )

        origin_preference = st.selectbox(
            "🌎 선호하는 카카오 원산지",
            ["상관없음"] + origin_list
        )

    bean_list = sorted(
        df["Bean_Type"].unique().tolist()
    )

    bean_preference = st.selectbox(
        "🌱 선호하는 카카오 원두 종류",
        ["상관없음"] + bean_list
    )

    st.markdown("---")

    if st.button(
        "🍫 나에게 맞는 초콜릿 추천받기",
        use_container_width=True
    ):

        # 전체 데이터에서 후보 선택
        candidates = df.copy()

        # 원산지 조건
        if origin_preference != "상관없음":
            candidates = candidates[
                candidates["Broad_Origin"] == origin_preference
            ]

        # 원두 종류 조건
        if bean_preference != "상관없음":
            candidates = candidates[
                candidates["Bean_Type"] == bean_preference
            ]

        # 조건에 맞는 데이터가 부족하면 원산지 조건만 적용
        if len(candidates) < 5:

            candidates = df.copy()

            if origin_preference != "상관없음":
                candidates = candidates[
                    candidates["Broad_Origin"] == origin_preference
                ]

        # 그래도 부족하면 전체 데이터 사용
        if len(candidates) < 5:
            candidates = df.copy()

            st.warning(
                "선택한 조건에 맞는 초콜릿이 부족하여 "
                "전체 데이터에서 추천합니다."
            )

        # AI 예상 평점 계산
        predicted = model.predict(
            candidates[feature_cols]
        )

        candidates["AI_예상평점"] = np.clip(
            predicted,
            0,
            5
        )

        # 원하는 카카오 함량과의 차이 계산
        candidates["함량_차이"] = abs(
            candidates["Cocoa_Percent"] -
            cocoa_preference
        )

        # AI 예상 평점과 취향 유사도를 함께 반영
        candidates["추천점수"] = (
            candidates["AI_예상평점"]
            - candidates["함량_차이"] * 0.025
        )

        # 추천 점수 순으로 정렬
        recommendations = candidates.sort_values(
            "추천점수",
            ascending=False
        ).head(5).copy()

        recommendations = recommendations.reset_index(
            drop=True
        )

        recommendations.index = (
            recommendations.index + 1
        )

        st.success("AI 초콜릿 추천이 완료되었습니다! 🎉")

        st.subheader("🏆 당신을 위한 초콜릿 TOP 5")

        for rank, row in recommendations.iterrows():

            with st.container(border=True):

                st.markdown(
                    f"### {rank}위 🏅 {row['Company']}"
                )

                st.write(
                    f"🍫 카카오 함량: {row['Cocoa_Percent']}%"
                )

                st.write(
                    f"🌎 원산지: {row['Broad_Origin']}"
                )

                st.write(
                    f"🌱 원두 종류: {row['Bean_Type']}"
                )

                st.write(
                    f"⭐ AI 예상 평점: "
                    f"{row['AI_예상평점']:.2f} / 5.00"
                )

                st.caption(
                    f"원두 정보: {row['Bean_Origin']}"
                )

        # 추천 결과 그래프
        st.markdown("---")

        st.subheader("📊 AI 추천 결과 비교")

        chart_data = recommendations[
            ["AI_예상평점"]
        ].copy()

        chart_data.index = [
            f"{i}위" for i in range(1, len(chart_data) + 1)
        ]

        st.bar_chart(chart_data)

        st.info(
            "추천 결과는 인공지능이 예상한 평점과 "
            "선택한 카카오 함량의 유사도를 함께 고려한 결과입니다."
        )

        st.caption(
            "※ AI 예상 평점은 실제 소비자 평가가 아닌 모델의 추정값입니다."
        )


# =====================================
# 7. 초콜릿 데이터 분석
# =====================================

elif menu == "📊 초콜릿 데이터 분석":

    st.header("📊 초콜릿 데이터 분석")

    st.subheader("카카오 함량과 평점의 관계")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        df["Cocoa_Percent"],
        df["Rating"],
        alpha=0.4
    )

    ax.set_xlabel("Cocoa Percent (%)")
    ax.set_ylabel("Rating")
    ax.set_title("Cocoa Percent vs Rating")
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("카카오 함량 구간별 평균 평점")

    df_chart = df.copy()

    df_chart["함량 구간"] = pd.cut(
        df_chart["Cocoa_Percent"],
        bins=[0, 50, 60, 70, 80, 90, 100],
        labels=[
            "50% 이하",
            "50~60%",
            "60~70%",
            "70~80%",
            "80~90%",
            "90~100%"
        ],
        include_lowest=True
    )

    group_rating = (
        df_chart.groupby(
            "함량 구간",
            observed=False
        )["Rating"]
        .mean()
        .dropna()
    )

    st.bar_chart(group_rating)


# =====================================
# 8. 인기 초콜릿 순위
# =====================================

elif menu == "🏆 인기 초콜릿 순위":

    st.header("🏆 평점이 높은 초콜릿 TOP 10")

    top10 = df.sort_values(
        "Rating",
        ascending=False
    ).head(10)

    result = top10[
        [
            "Company",
            "Bean_Origin",
            "Cocoa_Percent",
            "Broad_Origin",
            "Rating"
        ]
    ].copy()

    result.insert(
        0,
        "순위",
        range(1, len(result) + 1)
    )

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("⭐ 평점 비교")

    chart = top10.copy()

    chart["초콜릿"] = (
        chart["Company"].astype(str)
        + " - "
        + chart["Bean_Origin"].astype(str)
    )

    chart = chart.set_index("초콜릿")

    st.bar_chart(chart["Rating"])


# =====================================
# 9. AI 학습 결과
# =====================================

elif menu == "🤖 AI 학습 결과":

    st.header("🤖 인공지능 학습 결과")

    st.write(
        "Random Forest 회귀 알고리즘을 사용하여 "
        "초콜릿의 특징과 평점 사이의 관계를 학습했습니다."
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "평균 절대 오차 (MAE)",
        f"{mae:.3f}"
    )

    col2.metric(
        "결정계수 (R²)",
        f"{r2:.3f}"
    )

    st.markdown("---")

    st.subheader("🧠 인공지능 작동 과정")

    st.write("""
    1. 초콜릿 데이터 수집
    2. 결측값과 문자 데이터 처리
    3. 학습용 데이터와 테스트용 데이터 분리
    4. Random Forest 회귀 모델 학습
    5. 초콜릿의 예상 평점 계산
    6. 사용자의 취향과 예상 평점을 이용하여 초콜릿 추천
    """)

    st.info(
        "MAE는 예측값과 실제값의 평균적인 차이를 나타냅니다. "
        "R²는 모델이 데이터의 변동을 얼마나 설명하는지 나타냅니다."
    )

    st.caption(
        "모델의 성능 수치는 실제 데이터로 학습하고 테스트한 결과입니다."
    )


# =====================================
# 10. 하단
# =====================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#79452D;">
        <h3>🍫 AI Chocolate Recommend</h3>
        나만의 초콜릿을 인공지능으로 찾아보세요!
    </div>
    """,
    unsafe_allow_html=True
)
