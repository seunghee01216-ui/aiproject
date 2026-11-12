
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# ----------------------------------
# 1. 페이지 설정
# ----------------------------------

st.set_page_config(
    page_title="붓꽃 품종 예측 AI",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 AI 붓꽃 품종 예측 프로그램")

st.write(
    "꽃잎의 길이와 너비를 입력하면 "
    "AI가 붓꽃의 품종을 예측해 줍니다!"
)

st.divider()

# ----------------------------------
# 2. CSV 파일 불러오기
# ----------------------------------

@st.cache_data
def load_data():

    # pages 폴더의 상위 폴더에서 CSV 불러오기
    file_path = (
        Path(__file__).resolve().parent.parent
        / "Iris.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다: {file_path}"
        )

    try:
        data = pd.read_csv(
            file_path,
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError:
        data = pd.read_csv(
            file_path,
            encoding="cp949"
        )

    return data


try:
    df = load_data()

except Exception as e:
    st.error(f"CSV 파일을 불러오지 못했습니다: {e}")
    st.stop()

# ----------------------------------
# 3. 데이터 전처리
# ----------------------------------

df.columns = df.columns.astype(str).str.strip()

column_aliases = {
    "꽃잎길이": [
        "꽃잎길이",
        "PetalLengthCm",
        "petal_length"
    ],
    "꽃잎너비": [
        "꽃잎너비",
        "PetalWidthCm",
        "petal_width"
    ],
    "품종": [
        "품종",
        "Species",
        "species"
    ]
}

selected_columns = {}

for standard_name, aliases in column_aliases.items():

    found = next(
        (col for col in aliases if col in df.columns),
        None
    )

    if found is None:
        st.error(
            f"{standard_name} 열을 찾을 수 없습니다."
        )
        st.stop()

    selected_columns[standard_name] = found

data = df[
    [
        selected_columns["꽃잎길이"],
        selected_columns["꽃잎너비"],
        selected_columns["품종"]
    ]
].copy()

data.columns = [
    "꽃잎길이",
    "꽃잎너비",
    "품종"
]

data["꽃잎길이"] = pd.to_numeric(
    data["꽃잎길이"],
    errors="coerce"
)

data["꽃잎너비"] = pd.to_numeric(
    data["꽃잎너비"],
    errors="coerce"
)

data["품종"] = data["품종"].astype("string").str.strip()

data = data.dropna().copy()
data = data[data["품종"] != ""]

if len(data) == 0 or data["품종"].nunique() < 2:
    st.error("학습에 필요한 데이터가 부족합니다.")
    st.stop()

X = data[["꽃잎길이", "꽃잎너비"]]
y = data["품종"].astype(str)

# ----------------------------------
# 4. 입력값 범위 계산
# ----------------------------------

petal_length_min = float(data["꽃잎길이"].min())
petal_length_max = float(data["꽃잎길이"].max())

petal_width_min = float(data["꽃잎너비"].min())
petal_width_max = float(data["꽃잎너비"].max())

# ----------------------------------
# 5. 학습 데이터와 테스트 데이터 분리
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------
# 6. 랜덤 포레스트 모델 생성
# ----------------------------------

@st.cache_resource
def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=5
    )

    model.fit(X_train, y_train)

    return model


model = train_model(X_train, y_train)

# 정확도 계산
y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

# ----------------------------------
# 7. 알고리즘 설명
# ----------------------------------

st.subheader("🤖 사용 알고리즘 소개")

with st.expander(
    "🌳 랜덤 포레스트(Random Forest)란?",
    expanded=True
):

    st.write("""
    랜덤 포레스트는 여러 개의 결정 트리(Decision Tree)를
    만들어 결과를 종합하는 머신러닝 분류 알고리즘입니다.

    하나의 나무가 아니라 여러 개의 나무가 함께 판단하기 때문에
    랜덤 포레스트(Random Forest)라는 이름을 가지고 있습니다.
    """)

    st.markdown("### 🌳 작동 원리")

    st.write("""
    **1단계: 데이터 학습**

    Iris 데이터에서 꽃잎 길이와 꽃잎 너비를 학습합니다.

    **2단계: 여러 개의 결정 트리 생성**

    서로 다른 학습 데이터와 특성 조합을 이용하여
    여러 개의 결정 트리를 만듭니다.

    **3단계: 각각의 트리가 품종 예측**

    각 결정 트리가 입력된 꽃잎 크기를 보고
    어떤 품종인지 판단합니다.

    **4단계: 최종 결과 결정**

    여러 결정 트리의 투표 결과를 종합하여
    가장 많은 표를 얻은 품종을 최종 예측합니다.
    """)

    st.markdown("### ⚙️ 이번 프로그램의 모델 설정")

    st.write("""
    - 알고리즘: Random Forest Classifier
    - 결정 트리 개수: 200개
    - 트리의 최대 깊이: 5
    - 학습 데이터: 전체 데이터의 80%
    - 테스트 데이터: 전체 데이터의 20%
    """)

    st.markdown("### 👍 랜덤 포레스트의 장점")

    st.write("""
    - 여러 결정 트리의 결과를 종합하여 예측합니다.
    - 분류 문제에 폭넓게 활용됩니다.
    - 각 품종의 예측 확률을 확인할 수 있습니다.
    - 특성 중요도를 통해 어떤 데이터가 예측에
      더 많이 활용되었는지 확인할 수 있습니다.
    """)

st.divider()

# ----------------------------------
# 8. 모델 학습 결과
# ----------------------------------

st.subheader("📊 AI 모델 학습 결과")

col1, col2, col3 = st.columns(3)

col1.metric(
    "전체 데이터",
    f"{len(data)}개"
)

col2.metric(
    "학습 데이터",
    f"{len(X_train)}개"
)

col3.metric(
    "예측 정확도",
    f"{accuracy * 100:.1f}%"
)

st.caption(
    "정확도는 분리한 테스트 데이터에 대한 결과입니다."
)

# ----------------------------------
# 9. 특성 중요도
# ----------------------------------

st.subheader("🔍 AI가 중요하게 생각하는 데이터")

importance_df = pd.DataFrame({
    "특성": ["꽃잎 길이", "꽃잎 너비"],
    "중요도": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "중요도",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("특성"),
    y="중요도"
)

st.info(
    "특성 중요도는 모델의 예측 과정에서 각 특성이 "
    "얼마나 기여했는지를 나타내는 지표입니다. "
    "인과관계를 의미하지는 않습니다."
)

st.divider()

# ----------------------------------
# 10. 사용자 입력
# ----------------------------------

st.subheader("✏️ 꽃잎 크기 입력하기")

st.write(
    "아래 범위 안에서 꽃잎의 길이와 너비를 입력하세요."
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 🌷 꽃잎 길이")

    st.caption(
        f"입력 가능 범위: "
        f"{petal_length_min:.1f} ~ "
        f"{petal_length_max:.1f} cm"
    )

    petal_length = st.number_input(
        "꽃잎 길이 (cm)",
        min_value=petal_length_min,
        max_value=petal_length_max,
        value=1.5,
        step=0.1,
        format="%.1f"
    )

with col2:

    st.markdown("### 🌷 꽃잎 너비")

    st.caption(
        f"입력 가능 범위: "
        f"{petal_width_min:.1f} ~ "
        f"{petal_width_max:.1f} cm"
    )

    petal_width = st.number_input(
        "꽃잎 너비 (cm)",
        min_value=petal_width_min,
        max_value=petal_width_max,
        value=0.3,
        step=0.1,
        format="%.1f"
    )

# ----------------------------------
# 11. 품종 예측
# ----------------------------------

input_data = pd.DataFrame(
    [[petal_length, petal_width]],
    columns=["꽃잎길이", "꽃잎너비"]
)

prediction = model.predict(input_data)[0]

probabilities = model.predict_proba(input_data)[0]

st.divider()

st.subheader("🔮 AI 예측 결과")

st.success(
    f"🌸 예측된 붓꽃 품종: {prediction}"
)

st.write(
    f"꽃잎 길이: **{petal_length}cm**"
)

st.write(
    f"꽃잎 너비: **{petal_width}cm**"
)

# ----------------------------------
# 12. 품종별 예측 확률
# ----------------------------------

st.subheader("📈 품종별 예측 확률")

prob_df = pd.DataFrame({
    "품종": model.classes_,
    "예측 확률": probabilities
})

prob_df = prob_df.sort_values(
    "예측 확률",
    ascending=False
)

st.bar_chart(
    prob_df.set_index("품종"),
    y="예측 확률"
)

for _, row in prob_df.iterrows():

    st.write(
        f"🌼 {row['품종']}: "
        f"{row['예측 확률'] * 100:.1f}%"
    )

st.divider()

# ----------------------------------
# 13. 품종별 색상 설정
# ----------------------------------

st.subheader("📊 꽃잎 길이와 너비에 따른 품종 분포")

species = list(model.classes_)

color_list = [
    "#FF9999",
    "#66B3FF",
    "#99CC99"
]

colors = {
    name: color_list[i % len(color_list)]
    for i, name in enumerate(species)
}

cmap = ListedColormap([
    colors[name] for name in species
])

# ----------------------------------
# 14. AI 분류 영역 계산
# ----------------------------------

x_min = min(
    X["꽃잎길이"].min() - 0.5,
    petal_length - 0.5
)

x_max = max(
    X["꽃잎길이"].max() + 0.5,
    petal_length + 0.5
)

y_min = min(
    X["꽃잎너비"].min() - 0.3,
    petal_width - 0.3
)

y_max = max(
    X["꽃잎너비"].max() + 0.3,
    petal_width + 0.3
)

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid_data = pd.DataFrame({
    "꽃잎길이": xx.ravel(),
    "꽃잎너비": yy.ravel()
})

Z = model.predict(grid_data)

species_to_num = {
    name: i for i, name in enumerate(species)
}

Z_num = np.array([
    species_to_num[name] for name in Z
]).reshape(xx.shape)

# ----------------------------------
# 15. 그래프 시각화
# ----------------------------------

fig, ax = plt.subplots(figsize=(10, 7))

# AI의 품종별 분류 영역
ax.contourf(
    xx,
    yy,
    Z_num,
    levels=np.arange(len(species) + 1) - 0.5,
    cmap=cmap,
    alpha=0.25
)

# 실제 데이터 표시
for name in species:

    subset = data[data["품종"] == name]

    ax.scatter(
        subset["꽃잎길이"],
        subset["꽃잎너비"],
        color=colors[name],
        label=name,
        alpha=0.7,
        edgecolors="black",
        s=45
    )

# 사용자 입력값: 노란색 별
ax.scatter(
    petal_length,
    petal_width,
    color="yellow",
    marker="*",
    s=450,
    edgecolors="black",
    linewidths=1.5,
    label="YOUR INPUT",
    zorder=10
)

# 예측 결과 표시
ax.annotate(
    f"Predicted: {prediction}",
    (petal_length, petal_width),
    xytext=(12, 15),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
    bbox=dict(
        boxstyle="round",
        fc="white",
        alpha=0.9
    )
)

ax.set_xlabel(
    "꽃잎 길이 (cm)",
    fontsize=12
)

ax.set_ylabel(
    "꽃잎 너비 (cm)",
    fontsize=12
)

ax.set_title(
    "Iris Species Classification",
    fontsize=15
)

ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig)

plt.close(fig)

# ----------------------------------
# 16. 분류 이유 설명
# ----------------------------------

st.divider()

st.subheader("🔍 AI는 왜 이 품종으로 분류했을까?")

st.write("""
그래프의 배경색은 AI가 각 위치에서 예측하는
붓꽃의 품종을 나타냅니다.
""")

st.write(
    "⭐ 노란색 별은 사용자가 입력한 꽃잎의 위치입니다."
)

st.success(
    f"입력한 위치는 AI가 {prediction}으로 "
    "분류하는 영역에 있습니다."
)

st.markdown("### 🌈 품종별 분류 영역")

for name in species:

    st.markdown(
        f"<span style='color:{colors[name]};'>●</span> "
        f"**{name}**",
        unsafe_allow_html=True
    )

st.markdown("### 💡 예측 결과 해석")

best_probability = probabilities[
    list(model.classes_).index(prediction)
]

st.write(
    f"입력한 꽃잎 크기에 대해 "
    f"**{prediction}**의 예측 확률은 "
    f"**{best_probability * 100:.1f}%**입니다."
)

st.write("""
랜덤 포레스트는 여러 결정 트리의 투표 결과를
종합하여 최종 품종을 결정합니다.

따라서 그래프의 분류 영역과 품종별 예측 확률을
함께 확인하면 AI의 판단을 이해하는 데 도움이 됩니다.
""")

st.caption(
    "이 프로그램은 꽃잎 길이와 꽃잎 너비 "
    "두 가지 특성만 사용합니다."
)
