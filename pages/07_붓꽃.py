
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

    # pages 폴더의 상위 폴더에서 CSV 찾기
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
# 3. 데이터 열 확인 및 전처리
# ----------------------------------

# 실제 CSV의 열 이름에 맞게 공백 제거
df.columns = df.columns.astype(str).str.strip()

# 한글 또는 일반적인 영문 열 이름 지원
column_aliases = {
    "꽃잎길이": [
        "꽃잎길이", "PetalLengthCm", "petal_length"
    ],
    "꽃잎너비": [
        "꽃잎너비", "PetalWidthCm", "petal_width"
    ],
    "품종": [
        "품종", "Species", "species"
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
            f"'{standard_name}'에 해당하는 열을 찾을 수 없습니다. "
            f"현재 열 이름: {list(df.columns)}"
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

data.columns = ["꽃잎길이", "꽃잎너비", "품종"]

data["꽃잎길이"] = pd.to_numeric(
    data["꽃잎길이"], errors="coerce"
)

data["꽃잎너비"] = pd.to_numeric(
    data["꽃잎너비"], errors="coerce"
)

data["품종"] = data["품종"].astype("string").str.strip()

data = data.dropna().copy()
data = data[data["품종"] != ""]

if len(data) == 0 or data["품종"].nunique() < 2:
    st.error("학습에 필요한 데이터 또는 품종이 부족합니다.")
    st.stop()

X = data[["꽃잎길이", "꽃잎너비"]]
y = data["품종"].astype(str)

# ----------------------------------
# 4. 학습 데이터와 테스트 데이터 분리
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------
# 5. 랜덤 포레스트 모델 학습
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
accuracy = accuracy_score(y_test, y_pred)

# ----------------------------------
# 6. 모델 학습 결과
# ----------------------------------

st.subheader("🤖 AI 모델 학습 결과")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(data)}개")
col2.metric("학습 데이터", f"{len(X_train)}개")
col3.metric("예측 정확도", f"{accuracy * 100:.1f}%")

st.caption(
    "테스트 데이터에 대한 정확도이며 "
    "새로운 데이터의 정확도를 보장하지는 않습니다."
)

st.divider()

# ----------------------------------
# 7. 사용자 입력
# ----------------------------------

st.subheader("✏️ 꽃잎 크기 입력하기")

col1, col2 = st.columns(2)

with col1:
    petal_length = st.number_input(
        "🌷 꽃잎 길이 (cm)",
        min_value=0.1,
        max_value=10.0,
        value=1.5,
        step=0.1
    )

with col2:
    petal_width = st.number_input(
        "🌷 꽃잎 너비 (cm)",
        min_value=0.1,
        max_value=5.0,
        value=0.3,
        step=0.1
    )

# ----------------------------------
# 8. 품종 예측
# ----------------------------------

input_data = pd.DataFrame(
    [[petal_length, petal_width]],
    columns=["꽃잎길이", "꽃잎너비"]
)

prediction = model.predict(input_data)[0]

probabilities = model.predict_proba(input_data)[0]

st.divider()

st.subheader("🔮 AI 예측 결과")

st.success(f"🌸 예측된 붓꽃 품종: {prediction}")

st.write(
    f"꽃잎 길이: **{petal_length}cm** / "
    f"꽃잎 너비: **{petal_width}cm**"
)

st.write("### 품종별 예측 확률")

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

# ----------------------------------
# 9. 품종별 색상 설정
# ----------------------------------

st.divider()

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
# 10. AI 분류 영역 계산
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
# 11. 그래프 그리기
# ----------------------------------

fig, ax = plt.subplots(figsize=(10, 7))

# AI 분류 영역
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

# 사용자가 입력한 값 표시
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

# 예측 품종 표시
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

ax.set_xlabel("꽃잎 길이 (cm)", fontsize=12)
ax.set_ylabel("꽃잎 너비 (cm)", fontsize=12)

ax.set_title(
    "Iris Species Classification",
    fontsize=15
)

ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig)
plt.close(fig)

# ----------------------------------
# 12. 분류 이유 설명
# ----------------------------------

st.divider()

st.subheader("🔍 AI는 왜 이 품종으로 분류했을까?")

st.write(
    "그래프의 배경색은 랜덤 포레스트가 "
    "각 위치에서 예측하는 붓꽃 품종을 나타냅니다."
)

st.write(
    "⭐ 노란색 별은 사용자가 입력한 꽃잎의 위치입니다."
)

st.success(
    f"입력한 위치는 AI의 분류 영역 중 "
    f"{prediction} 영역에 해당합니다."
)

st.write("### 🌈 품종별 분류 영역")

for name in species:

    st.markdown(
        f"<span style='color:{colors[name]};'>●</span> "
        f"**{name}**",
        unsafe_allow_html=True
    )

st.info(
    "랜덤 포레스트는 여러 결정 트리의 예측을 종합하여 "
    "최종 품종을 결정합니다. 그래프의 배경색은 "
    "모델이 학습한 분류 영역을 시각화한 것입니다."
)

st.caption(
    "본 프로그램은 꽃잎 길이와 꽃잎 너비 "
    "두 가지 특성을 사용하여 붓꽃 품종을 예측합니다."
)
