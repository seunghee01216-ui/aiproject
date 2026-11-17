
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="AI Chocolate Lab",
    page_icon="🍫",
    layout="wide"
)

st.title("🍫 AI Chocolate Lab")
st.subheader("인공지능으로 분석하는 초콜릿 연구소")
st.write("초콜릿 데이터를 분석하고 인공지능으로 평점을 예측해 보세요!")

st.markdown("---")

# -----------------------------
# 데이터 파일 불러오기
# -----------------------------
@st.cache_data
def load_data():

    folder = Path(__file__).resolve().parent

    possible_paths = [
        folder / "flavors_of_cacao.csv",
        folder.parent / "flavors_of_cacao.csv"
    ]

    # 업로드 파일명이 flavors_of_cacao[1].csv인 경우도 탐색
    possible_paths += list(folder.glob("flavors_of_cacao*.csv"))
    possible_paths += list(folder.parent.glob("flavors_of_cacao*.csv"))

    file_path = next(
        (p for p in possible_paths if p.exists()),
        None
    )

    if file_path is None:
        raise FileNotFoundError(
            "flavors_of_cacao.csv 파일을 찾을 수 없습니다."
        )

    data = pd.read_csv(file_path)

    # 열 이름 정리
    data.columns = [
        str(c).replace("\xa0", " ").replace("\n", " ").strip()
        for c in data.columns
    ]

    data = data.rename(columns={
        "Company (Maker-if known)": "Company",
        "Specific Bean Origin or Bar Name": "Bean_Origin",
        "REF": "REF",
        "Review Date": "Review_Date",
        "Cocoa Percent": "Cocoa_Percent",
        "Company Location": "Company_Location",
        "Rating": "Rating",
        "Bean Type": "Bean_Type",
        "Broad Bean Origin": "Broad_Origin"
    })

    # 카카오 함량 숫자로 변환
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

    # 문자열 데이터의 공백 처리
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype("string").str.strip()
        data[col] = data[col].replace("", pd.NA)

    return data


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터 파일을 불러오지 못했습니다: {e}")
    st.stop()


# -----------------------------
# 데이터 전처리
# -----------------------------
df = df.dropna(subset=["Cocoa_Percent", "Rating"])

df = df[
    (df["Cocoa_Percent"] >= 0) &
    (df["Cocoa_Percent"] <= 100) &
    (df["Rating"] >= 0) &
    (df["Rating"] <= 5)
].copy()

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

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

# -----------------------------
# 인공지능 모델 학습
# -----------------------------
@st.cache_resource
def train_model(data):

    features = categorical_cols + numeric_cols

    X = data[features]
    y = data["Rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 범주형 데이터 처리
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 숫자형 데이터 처리
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer([
        ("cat", categorical_transformer, categorical_cols),
        ("num", numeric_transformer, numeric_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mae, r2


model, mae, r2 = train_model(df)

# -----------------------------
# 사이드바 메뉴
# -----------------------------
menu = st.sidebar.radio(
    "🍫 메뉴 선택",
    [
        "🏠 홈",
        "🧠 AI 평점 예측",
        "📊 카카오 함량 분석",
        "🌎 국가별 분석",
        "🏆 초콜릿 순위"
    ]
)

# -----------------------------
# 홈 화면
# -----------------------------
if menu == "🏠 홈":

    st.header("🍫 초콜릿 데이터 분석")

    col1, col2, col3 = st.columns(3)

    col1.metric("초콜릿 데이터 수", f"{len(df):,}개")
    col2.metric(
        "평균 카카오 함량",
        f"{df['Cocoa_Percent'].mean():.1f}%"
    )
    col3.metric(
        "평균 평점",
        f"{df['Rating'].mean():.2f}점"
    )

    st.markdown("---")

    st.subheader("📋 데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("🤖 인공지능 모델 성능")

    col1, col2 = st.columns(2)
    col1.metric("평균 절대 오차(MAE)", f"{mae:.3f}")
    col2.metric("결정계수(R²)", f"{r2:.3f}")

    st.info(
        "인공지능은 기존 초콜릿 데이터를 학습하여 "
        "새로운 초콜릿의 예상 평점을 예측합니다."
    )

# -----------------------------
# AI 평점 예측
# -----------------------------
elif menu == "🧠 AI 평점 예측":

    st.header("🧠 AI 초콜릿 평점 예측")
    st.write("초콜릿의 정보를 입력하면 예상 평점을 알려줍니다.")

    col1, col2 = st.columns(2)

    with col1:

        cocoa = st.slider(
            "카카오 함량 (%)",
            min_value=40,
            max_value=100,
            value=70
        )

        company = st.selectbox(
            "제조 회사",
            sorted(df["Company"].dropna().unique())
        )

        bean_origin = st.selectbox(
            "초콜릿 원두 종류",
            sorted(df["Bean_Origin"].dropna().unique())
        )

    with col2:

        bean_type = st.selectbox(
            "카카오 원두 유형",
            sorted(df["Bean_Type"].dropna().unique())
        )

        broad_origin = st.selectbox(
            "카카오 원산지",
            sorted(df["Broad_Origin"].dropna().unique())
        )

        review_year = st.number_input(
            "평가 연도",
            min_value=2000,
            max_value=2030,
            value=2016
        )

    if st.button("🍫 예상 평점 확인하기", use_container_width=True):

        new_data = pd.DataFrame([{
            "Company": company,
            "Bean_Origin": bean_origin,
            "Bean_Type": bean_type,
            "Broad_Origin": broad_origin,
            "Cocoa_Percent": cocoa,
            "Review_Date": review_year
        }])

        prediction = model.predict(new_data)[0]
        prediction = np.clip(prediction, 0, 5)

        st.success(f"예측 평점: ⭐ {prediction:.2f} / 5.00")

        if prediction >= 3.5:
            st.write("🏆 인공지능이 높은 평점을 예상했어요!")
        elif prediction >= 2.5:
            st.write("🍫 보통 수준의 평점이 예상돼요.")
        else:
            st.write("📉 상대적으로 낮은 평점이 예상돼요.")

        st.caption(
            "※ 실제 소비자 평가가 아닌 학습 데이터에 기반한 모델의 예측값입니다."
        )

# -----------------------------
# 카카오 함량 분석
# -----------------------------
elif menu == "📊 카카오 함량 분석":

    st.header("📊 카카오 함량과 초콜릿 평점의 관계")

    st.write(
        "카카오 함량이 달라지면 초콜릿 평점은 어떻게 변할까요?"
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        df["Cocoa_Percent"],
        df["Rating"],
        alpha=0.35
    )

    ax.set_xlabel("Cocoa Percent (%)")
    ax.set_ylabel("Rating")
    ax.set_title("Cocoa Percent vs Rating")
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    # 카카오 함량 구간별 평균 평점
    df["Cocoa_Group"] = pd.cut(
        df["Cocoa_Percent"],
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
        df.groupby("Cocoa_Group", observed=False)["Rating"]
        .mean()
        .dropna()
    )

    st.subheader("카카오 함량 구간별 평균 평점")

    st.bar_chart(group_rating)

    st.dataframe(
        group_rating.rename("평균 평점").reset_index(),
        use_container_width=True
    )

# -----------------------------
# 국가별 분석
# -----------------------------
elif menu == "🌎 국가별 분석":

    st.header("🌎 국가별 초콜릿 분석")

    countries = sorted(
        df["Broad_Origin"].dropna().unique()
    )

    selected_country = st.selectbox(
        "카카오 원산지를 선택하세요",
        countries
    )

    country_df = df[
        df["Broad_Origin"] == selected_country
    ]

    st.subheader(f"🌎 {selected_country} 초콜릿 분석")

    col1, col2, col3 = st.columns(3)

    col1.metric("제품 수", f"{len(country_df)}개")
    col2.metric(
        "평균 평점",
        f"{country_df['Rating'].mean():.2f}점"
    )
    col3.metric(
        "평균 카카오 함량",
        f"{country_df['Cocoa_Percent'].mean():.1f}%"
    )

    st.subheader("평점 분포")

    st.bar_chart(
        country_df["Rating"].value_counts().sort_index()
    )

    st.subheader("해당 국가 초콜릿 데이터")
    st.dataframe(
        country_df[
            [
                "Company",
                "Bean_Origin",
                "Cocoa_Percent",
                "Rating"
            ]
        ],
        use_container_width=True
    )

# -----------------------------
# 초콜릿 순위
# -----------------------------
elif menu == "🏆 초콜릿 순위":

    st.header("🏆 최고 평점 초콜릿 TOP 10")

    top10 = df.sort_values(
        "Rating",
        ascending=False
    ).head(10)

    st.dataframe(
        top10[
            [
                "Company",
                "Bean_Origin",
                "Cocoa_Percent",
                "Broad_Origin",
                "Rating"
            ]
        ].reset_index(drop=True),
        use_container_width=True
    )

    st.subheader("⭐ 평점 TOP 10 그래프")

    chart_data = top10.copy()
    chart_data["초콜릿"] = (
        chart_data["Company"].astype(str)
        + " - "
        + chart_data["Bean_Origin"].astype(str)
    )

    chart_data = chart_data.set_index("초콜릿")

    st.bar_chart(chart_data["Rating"])

    st.caption(
        "※ 동일한 평점을 가진 제품이 있을 수 있으며, "
        "개별 데이터 행의 평점을 기준으로 정렬했습니다."
    )

# -----------------------------
# 하단 설명
# -----------------------------
st.markdown("---")
st.caption("🍫 AI Chocolate Lab | Flavors of Cacao 데이터 활용")
st.caption("인공지능 모델: Random Forest Regression")

