"""Streamlit 기반 유방 종양 양성·악성 분류 웹 애플리케이션."""

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from main import FEATURES, load_project_data


st.set_page_config(
    page_title="유방 종양 분류기",
    page_icon="🔬",
    layout="centered",
)


@st.cache_resource
def train_model() -> tuple[Pipeline, float, float]:
    """기존 main.py와 같은 조건으로 모델을 학습한다."""
    x, y = load_project_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    return model, accuracy, roc_auc


st.markdown(
    """
    <style>
    .stApp { background: #f4f7f5; }
    .block-container { max-width: 760px; padding-top: 2.5rem; }
    .hero {
        padding: 28px 30px;
        border-radius: 20px;
        color: white;
        background: linear-gradient(135deg, #125d5b, #26877c);
        box-shadow: 0 14px 35px rgba(18, 93, 91, 0.18);
        margin-bottom: 22px;
    }
    .hero h1 { margin: 0 0 8px; font-size: 34px; }
    .hero p { margin: 0; color: #dcefed; line-height: 1.65; }
    .result-safe, .result-danger {
        padding: 22px;
        border-radius: 14px;
        text-align: center;
        margin-top: 16px;
    }
    .result-safe { background: #e5f5ee; border: 1px solid #acd9c8; color: #17664f; }
    .result-danger { background: #fbe9e7; border: 1px solid #e6bbb7; color: #9b3e38; }
    .result-safe h2, .result-danger h2 { margin: 0 0 8px; }
    .probability { font-size: 28px; font-weight: 800; }
    .notice {
        margin-top: 20px; padding: 13px 15px; border-radius: 10px;
        background: #eef1ef; color: #64726e; font-size: 13px;
    }
    </style>
    <div class="hero">
        <h1>유방 종양 분류기</h1>
        <p>세포핵의 평균 반지름, 평균 오목함, 평균 대칭성을 입력하면<br>
        로지스틱 회귀 모델이 양성·악성 가능성을 계산합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, accuracy, roc_auc = train_model()

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("사용 변수", "3개")
metric_col2.metric("정확도", f"{accuracy * 100:.2f}%")
metric_col3.metric("ROC-AUC", f"{roc_auc:.4f}")

st.subheader("세포핵 측정값 입력")
st.caption("아래 값은 일반적인 종양 크기가 아니라 검사 이미지에서 추출한 세포핵 특성값입니다.")

with st.form("prediction_form"):
    mean_radius = st.number_input(
        "평균 반지름 (mean radius)",
        min_value=0.0,
        value=14.13,
        step=0.01,
        format="%.4f",
        help="세포핵 중심에서 경계까지 거리의 평균",
    )
    mean_concavity = st.number_input(
        "평균 오목함 (mean concavity)",
        min_value=0.0,
        value=0.0888,
        step=0.001,
        format="%.4f",
        help="세포핵 경계가 안쪽으로 들어간 정도",
    )
    mean_symmetry = st.number_input(
        "평균 대칭성 (mean symmetry)",
        min_value=0.0,
        value=0.1812,
        step=0.001,
        format="%.4f",
        help="세포핵 형태의 평균 대칭성",
    )
    submitted = st.form_submit_button("분류 결과 확인", use_container_width=True)

if submitted:
    sample = pd.DataFrame(
        [[mean_radius, mean_concavity, mean_symmetry]],
        columns=FEATURES,
    )
    malignant_probability = float(model.predict_proba(sample)[0, 1])
    malignant = malignant_probability >= 0.5

    result_class = "result-danger" if malignant else "result-safe"
    result_title = "악성 가능성 높음 (Malignant)" if malignant else "양성 가능성 높음 (Benign)"
    st.markdown(
        f"""
        <div class="{result_class}">
            <h2>{result_title}</h2>
            <div>악성 확률</div>
            <div class="probability">{malignant_probability * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(malignant_probability, text="악성 확률")

st.markdown(
    """
    <div class="notice">
        <strong>교육용 분류 모델</strong><br>
        이 결과는 공개 데이터로 학습한 예측값이며 실제 의료 진단을 대신하지 않습니다.
    </div>
    """,
    unsafe_allow_html=True,
)
