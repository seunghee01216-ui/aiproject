
import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# ==================================
# 1. 페이지 설정
# ==================================

st.set_page_config(
    page_title="AI 초콜릿 추천",
    page_icon="🍬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #FFF9F2;
}
h1, h2, h3 {
    color: #75452E;
}
.stButton > button {
    background-color: #75452E;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

st.title("🍬 AI 초콜릿 추천")
st.write(
    "나의 초콜릿 취향을 선택하면 "
    "인공지능이 취향에 맞는 초콜릿 TOP 5를 추천합니다."
)

st.divider()


# ==================================
# 2. 데이터 불러오기
# ==================================


@st.cache_data
def load_data():

    folder = Path(__file__).resolve().parent
    parent_folder = folder.parent

    csv_path = parent_folder / "flavors_of_cacao.csv"

    if not csv_path.exists():
        st.error("CSV 파일을 찾을 수 없습니다.")
        st.stop()

    data = pd.read_csv(csv_path)

   


    # 열 이름 자동 정리
    def normalize(name):
        return re.sub(
            r"[^a-z0-9]",
            "",
            str(name).lower()
        )

    mapping = {
        "companymakerifknown": "Company",
        "specificbeanoriginorbarname": "Bean_Origin",
        "reviewdate": "Review_Date",
        "cocoapercent": "Cocoa_Percent",
        "beantype": "Bean_Type",
        "broadbeanorigin": "Broad_Origin",
        "rating": "Rating"
    }

    rename = {}

    for col in data.columns:
        key = normalize(col)

        if key in mapping:
            rename[col] = mapping[key]

    data = data.rename(columns=rename)

    required = [
        "Company",
        "Bean_Origin",
        "Review_Date",
        "Cocoa_Percent",
        "Bean_Type",
        "Broad_Origin",
        "Rating"
    ]

    for col in required:
        if col not in data.columns:
            data[col] = np.nan

    # 숫자 데이터 변환
    data["Cocoa_Percent"] = pd.to_numeric(
        data["Cocoa_Percent"]
        .astype(str)
        .str.replace("%", "", regex=False),
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

    data = data.dropna(
        subset=["Cocoa_Percent", "Rating"]
    )

    data = data[
        (data["Cocoa_Percent"].between(0, 100)) &
        (data["Rating"].between(0, 5))
    ].copy()

    # 문자 데이터 결측치 처리
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
            .replace(["", "nan", "<NA>"], pd.NA)
            .fillna("정보 없음")
        )

    return data


try:
    df = load_data()

except Exception as e:
    st.error(f"데이터 오류: {e}")
    st.stop()


# ==================================
# 3. 인공지능 학습
# ==================================

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

features = categorical_cols + numeric_cols


@st.cache_resource
def train_model(data):

    X = data[features]
    y = data["Rating"]

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocess = ColumnTransformer([
        ("cat", categorical, categorical_cols),
        ("num", numeric, numeric_cols)
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X, y)

    return model


if len(df) < 10:
    st.error("학습할 데이터가 부족합니다.")
    st.stop()

model = train_model(df)


# ==================================
# 4. 사용자 취향 선택
# ==================================

st.subheader("💝 나의 초콜릿 취향 선택")

col1, col2 = st.columns(2)

with col1:

    cocoa = st.slider(
        "🍫 원하는 카카오 함량 (%)",
        40, 100, 70
    )

    origin_options = sorted(
        df["Broad_Origin"].unique()
    )

    origin = st.selectbox(
        "🌎 선호하는 카카오 원산지",
        ["상관없음"] + origin_options
    )

with col2:

    bean_options = sorted(
        df["Bean_Type"].unique()
    )

    bean = st.selectbox(
        "🌱 선호하는 카카오 원두 종류",
        ["상관없음"] + bean_options
    )

st.caption(
    f"선택한 카카오 함량: {cocoa}%"
)

st.divider()


# ==================================
# 5. AI 초콜릿 추천
# ==================================

if st.button(
    "🍬 AI 초콜릿 추천받기",
    use_container_width=True
):

    candidates = df.copy()

    # 원산지 조건
    if origin != "상관없음":
        candidates = candidates[
            candidates["Broad_Origin"] == origin
        ]

    # 원두 종류 조건
    if bean != "상관없음":
        candidates = candidates[
            candidates["Bean_Type"] == bean
        ]

    # 조건에 맞는 데이터가 부족하면 조건 완화
    if len(candidates) < 5:

        st.warning(
            "선택한 조건에 맞는 제품이 적어 "
            "원산지 조건을 우선 적용합니다."
        )

        candidates = df.copy()

        if origin != "상관없음":
            candidates = candidates[
                candidates["Broad_Origin"] == origin
            ]

    if len(candidates) < 5:

        st.warning(
            "조건에 맞는 제품이 부족하여 "
            "전체 데이터에서 추천합니다."
        )

        candidates = df.copy()

    # AI 예상 평점
    candidates = candidates.copy()

    candidates["예상평점"] = np.clip(
        model.predict(candidates[features]),
        0,
        5
    )

    # 카카오 함량 취향과의 차이
    candidates["함량차이"] = abs(
        candidates["Cocoa_Percent"] - cocoa
    )

    # 추천 점수 계산
    candidates["추천점수"] = (
        candidates["예상평점"]
        - candidates["함량차이"] * 0.025
    )

    # 추천 TOP 5
    result = candidates.sort_values(
        "추천점수",
        ascending=False
    ).head(5)

    st.success("나에게 맞는 초콜릿을 찾았어요! 🎉")

    st.header("🏆 나만의 초콜릿 TOP 5")

    for i, (_, row) in enumerate(
        result.iterrows(), start=1
    ):

        with st.container(border=True):

            st.subheader(
                f"{i}위 🏅 {row['Company']}"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    f"🍫 카카오 함량: {row['Cocoa_Percent']}%"
                )

                st.write(
                    f"🌎 원산지: {row['Broad_Origin']}"
                )

            with col2:
                st.write(
                    f"🌱 원두 종류: {row['Bean_Type']}"
                )

                st.write(
                    f"⭐ AI 예상 평점: {row['예상평점']:.2f}점"
                )

            st.caption(
                f"초콜릿 이름: {row['Bean_Origin']}"
            )

    # ==================================
    # 6. 추천 결과 그래프
    # ==================================

    st.divider()

    st.subheader("📊 추천 초콜릿 평점 비교")

    chart = result[
        ["예상평점"]
    ].copy()

    chart.index = [
        f"{i}위" for i in range(1, len(chart) + 1)
    ]

    st.bar_chart(chart)

    st.info(
        "인공지능은 기존 초콜릿의 데이터를 학습하여 "
        "예상 평점을 계산하고, 카카오 함량 취향까지 "
        "반영하여 초콜릿을 추천합니다."
    )

    st.caption(
        "※ AI 예상 평점은 실제 평점이 아닌 모델의 예측값입니다."
    )


# ==================================
# 7. 하단
# ==================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;">
        <h3>🍬 AI Chocolate Recommend</h3>
        <p>나만의 초콜릿을 인공지능으로 찾아보세요!</p>
    </div>
    """,
    unsafe_allow_html=True
)
