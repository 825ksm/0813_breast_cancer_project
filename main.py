"""세포핵의 크기와 형태 특성을 이용한 유방암 진단 분류 프로젝트.

사용 변수
1. mean radius: 세포핵의 평균 반지름
2. mean concavity: 세포핵 경계의 오목한 정도
3. mean symmetry: 세포핵의 평균 대칭성
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["mean radius", "mean concavity", "mean symmetry"]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_project_data() -> tuple[pd.DataFrame, pd.Series]:
    """Wisconsin 유방암 데이터를 불러오고 악성=1, 양성=0으로 변환한다."""
    dataset = load_breast_cancer(as_frame=True)
    x = dataset.data[FEATURES].copy()
    y = (dataset.target == 0).astype(int)
    y.name = "malignant"
    return x, y


def save_feature_distributions(x: pd.DataFrame, y: pd.Series) -> None:
    """양성·악성에 따른 세 변수의 분포를 저장한다."""
    plot_data = x.copy()
    plot_data["diagnosis"] = y.map({0: "Benign", 1: "Malignant"})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for feature, axis in zip(FEATURES, axes):
        sns.boxplot(data=plot_data, x="diagnosis", y=feature, ax=axis)
        axis.set_title(feature)
        axis.set_xlabel("")
    fig.suptitle("Feature distributions by diagnosis")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_distributions.png", dpi=150)
    plt.close(fig)


def predict_new_sample(model: Pipeline) -> None:
    """사용자가 입력한 세포핵 특성으로 악성 확률을 예측한다."""
    print("\n=== New Sample Prediction ===")
    print("세포핵 측정값 3개를 입력하세요. 종료하려면 Enter만 누르세요.")

    while True:
        values = {}
        try:
            first_value = input("mean radius: ").strip()
            if not first_value:
                print("예측 입력을 종료합니다.")
                return

            values["mean radius"] = float(first_value)
            values["mean concavity"] = float(input("mean concavity: ").strip())
            values["mean symmetry"] = float(input("mean symmetry: ").strip())
        except ValueError:
            print("숫자만 입력해야 합니다. 다시 입력하세요.\n")
            continue

        sample = pd.DataFrame([values], columns=FEATURES)
        malignant_probability = model.predict_proba(sample)[0, 1]
        prediction = int(model.predict(sample)[0])
        diagnosis = "악성(Malignant)" if prediction == 1 else "양성(Benign)"

        print(f"예측 결과   : {diagnosis}")
        print(f"악성 확률   : {malignant_probability * 100:.2f}%")
        print("※ 교육용 예측 결과이며 실제 의료 진단에 사용할 수 없습니다.\n")

        again = input("다른 값을 예측할까요? (y/n): ").strip().lower()
        if again != "y":
            return


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
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

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_malignant": precision_score(y_test, predictions),
        "recall_malignant": recall_score(y_test, predictions),
        "f1_malignant": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    print("=== Breast Cancer Classification Results ===")
    print(f"Train samples: {len(x_train)}")
    print(f"Test samples : {len(x_test)}")
    for name, value in metrics.items():
        print(f"{name:20s}: {value:.4f}")

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Benign", "Malignant"],
            digits=4,
        )
    )

    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "metrics.csv", index=False)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Benign", "Malignant"],
        cmap="Blues",
    )
    display.ax_.set_title("Confusion Matrix")
    display.figure_.tight_layout()
    display.figure_.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(display.figure_)

    save_feature_distributions(x, y)
    print(f"Results saved to: {RESULTS_DIR}")

    predict_new_sample(model)


if __name__ == "__main__":
    main()
