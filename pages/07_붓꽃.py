
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
    file_path = Path(__file__).resolve().parent / "Iris.csv"

    try:
        data = pd.read_csv(file_path, encoding="cp949")
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding="utf-8-sig")

    return data

try:
    df = load_data()
except Exception as e:
    st.error(f"CSV 파일을 불러오지 못했습니다: {e}")
    st.stop()

# 필요한 열 확인
required_columns = ["꽃잎길이", "꽃잎너비", "품종"]

if not all(col in df.columns for col in required_columns):
    st.error("CSV 파일에 꽃잎길이, 꽃잎너비, 품종 열이 필요합니다.")
    st.stop()

# ----------------------------------
# 3. 데이터 전처리
# ----------------------------------

df = df[required_columns].copy()

df["꽃잎길이"] = pd.to_numeric(
    df["꽃잎길이"], errors="coerce"
)

df["꽃잎너비"] = pd.to_numeric(
    df["꽃잎너비"], errors="coerce"
)

df = df.dropna()

X = df[["꽃잎길이", "꽃잎너비"]]
y = df["품종"].astype(str)

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
# 5. 랜덤 포레스트 모델 생성
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
# 6. 모델 성능 표시
# ----------------------------------

st.subheader("🤖 AI 모델 학습 결과")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(df)}개")
col2.metric("학습 데이터", f"{len(X_train)}개")
col3.metric("예측 정확도", f"{accuracy * 100:.1f}%")

st.caption(
    "정확도는 테스트 데이터 30개에 대한 결과이며, "
    "새로운 데이터에서의 정확도를 보장하지는 않습니다."
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
    f"입력한 꽃잎 길이는 **{petal_length}cm**, "
    f"꽃잎 너비는 **{petal_width}cm**입니다."
)

st.write("### 품종별 예측 확률")

prob_df = pd.DataFrame({
    "품종": model.classes_,
    "예측 확률": probabilities
})

prob_df = prob_df.sort_values(
    "예측 확률", ascending=False
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
# 9. 시각화 준비
# ----------------------------------

st.divider()

st.subheader("📊 꽃잎 길이와 너비에 따른 품종 분포")

# 각 품종별 색상
species = sorted(y.unique())

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

# 그래프 범위
x_min = min(X["꽃잎길이"].min() - 0.5,
            petal_length - 0.5)

x_max = max(X["꽃잎길이"].max() + 0.5,
            petal_length + 0.5)

y_min = min(X["꽃잎너비"].min() - 0.3,
            petal_width - 0.3)

y_max = max(X["꽃잎너비"].max() + 0.3,
            petal_width + 0.3)

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid_data = pd.DataFrame({
    "꽃잎길이": xx.ravel(),
    "꽃잎너비": yy.ravel()
})

# 각 위치의 예측 품종
Z = model.predict(grid_data)

# 품종을 숫자로 변환
species_to_num = {
    name: i for i, name in enumerate(model.classes_)
}

Z_num = np.array([
    species_to_num[name] for name in Z
]).reshape(xx.shape)

# ----------------------------------
# 10. 그래프 그리기
# ----------------------------------

fig, ax = plt.subplots(figsize=(10, 7))

# AI의 분류 영역
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
    subset = df[df["품종"] == name]

    ax.scatter(
        subset["꽃잎길이"],
        subset["꽃잎너비"],
        color=colors[name],
        label=name,
        alpha=0.7,
        edgecolors="black",
        s=45
    )

# 사용자가 입력한 값: 노란색 별
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

# 입력값 옆에 예측 결과 표시
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

ax.set_xlabel("Petal Length (cm)", fontsize=12)
ax.set_ylabel("Petal Width (cm)", fontsize=12)

ax.set_title(
    "Iris Species Classification",
    fontsize=15
)

ax.legend()
ax.grid(alpha=0.2)

st.pyplot(fig)
plt.close(fig)

# ----------------------------------
# 11. 분류 이유 설명
# ----------------------------------

st.divider()

st.subheader("🔍 AI는 왜 이 품종으로 분류했을까?")

st.write(
    "그래프의 배경색은 랜덤 포레스트가 "
    "각 위치에서 예측하는 품종을 나타냅니다."
)

st.write(
    "⭐ 노란색 별은 사용자가 입력한 꽃잎의 위치입니다."
)

st.write(
    f"입력한 위치가 AI의 분류 영역 중 "
    f"**{prediction} 영역**에 있으므로 "
    f"해당 품종으로 예측되었습니다."
)

st.write("### 🌈 품종별 분류 영역")

for name in species:
    st.markdown(
        f"<span style='color:{colors[name]};'>"
        f"●</span> **{name}**",
        unsafe_allow_html=True
    )

st.info(
    "참고: 배경색은 모델의 분류 영역을 보여주며, "
    "실제 예측은 랜덤 포레스트의 여러 결정 트리 투표를 "
    "종합하여 이루어집니다."
)

st.caption(
    "본 프로그램은 Iris 데이터셋의 꽃잎 길이와 너비 "
    "두 가지 특성만 사용하여 품종을 예측합니다."
)
