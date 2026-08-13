# 세포핵 크기와 형태를 이용한 유방암 진단 분류

## 1. 프로젝트 개요

이 프로젝트는 세침흡인검사 이미지에서 추출된 세포핵 특성을 이용하여 유방 종양을 양성(Benign)과 악성(Malignant)으로 분류한다. 복잡한 전체 변수를 모두 사용하지 않고, 의미를 설명하기 쉬운 세 가지 변수만 사용하여 간단한 분류 모델을 구현하는 것이 목적이다.

> 이 데이터의 `mean radius`는 환자에게서 측정한 종양 전체의 실제 크기가 아니라 세포핵 이미지에서 계산한 평균 반지름이다.

## 2. 사용 데이터

- 데이터: Breast Cancer Wisconsin (Diagnostic)
- 표본 수: 569개
- 분류 대상: 양성 357개, 악성 212개
- 출처: scikit-learn의 `load_breast_cancer()`
- 원본: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## 3. 사용 변수

| 변수 | 구분 | 의미 |
|---|---|---|
| `mean radius` | 크기 | 세포핵 중심에서 경계까지 거리의 평균 |
| `mean concavity` | 형태 | 세포핵 경계가 안쪽으로 오목하게 들어간 정도 |
| `mean symmetry` | 형태 | 세포핵 모양의 평균 대칭성 |

## 4. 분석 방법

1. Wisconsin 유방암 데이터 불러오기
2. 세 개의 입력 변수 선택
3. 악성을 1, 양성을 0으로 변환
4. 데이터를 훈련용 80%, 평가용 20%로 분리
5. `StandardScaler`를 이용한 변수 표준화
6. 로지스틱 회귀 모델 학습
7. 정확도, 정밀도, 재현율, F1-score, ROC-AUC 평가
8. 혼동행렬과 변수별 분포 그래프 저장
9. 사용자가 입력한 세 가지 측정값의 양성·악성 및 악성 확률 예측

데이터 분할에는 `random_state=42`와 계층적 분할을 사용하여 실행할 때마다 같은 결과가 나오도록 했다.

## 5. 실행 방법

터미널에서 프로젝트 폴더로 이동한 후 다음 명령어를 입력한다.

```bash
pip install -r requirements.txt
python main.py
```

Windows에서 `python` 명령어가 동작하지 않으면 다음 명령어를 사용한다.

```bash
py main.py
```

### 웹 애플리케이션 실행

VS Code 터미널에서 다음 명령어를 입력한다.

```bash
streamlit run app.py
```

명령어가 인식되지 않으면 다음과 같이 실행한다.

```bash
py -m streamlit run app.py
```

실행 후 브라우저에서 `http://localhost:8501` 주소가 자동으로 열린다. 웹 화면에 `mean radius`, `mean concavity`, `mean symmetry`를 입력하고 **분류 결과 확인** 버튼을 누르면 양성·악성 결과와 악성 확률을 확인할 수 있다. 터미널을 종료하면 로컬 웹 애플리케이션도 종료된다.

## 6. 실행 결과

실행하면 터미널에 다음 평가 결과가 출력된다.

- Accuracy: 전체 예측 중 올바르게 분류한 비율
- Precision: 악성으로 예측한 표본 중 실제 악성의 비율
- Recall: 실제 악성 표본 중 악성으로 찾아낸 비율
- F1-score: 정밀도와 재현율의 조화평균
- ROC-AUC: 양성과 악성을 구분하는 전체적인 능력

`results` 폴더에는 다음 파일이 생성된다.

- `metrics.csv`: 평가 지표
- `confusion_matrix.png`: 혼동행렬
- `feature_distributions.png`: 양성·악성별 변수 분포

성능평가가 끝나면 터미널에 다음 세 값을 직접 입력할 수 있다.

```text
mean radius: 18.0
mean concavity: 0.20
mean symmetry: 0.22
```

프로그램은 입력값을 학습된 로지스틱 회귀 모델에 적용하여 양성·악성 분류와 악성 확률을 출력한다. 입력값은 일반 사진에서 얻는 값이 아니라 세침흡인검사 이미지에서 측정된 세포핵 특성값이어야 한다. 예측 기능은 교육용이며 실제 의료 진단에 사용할 수 없다.

평가용 데이터 구성에 따라 값은 달라질 수 있지만, 세 변수만 사용한 로지스틱 회귀 모델은 약 90% 수준의 정확도를 기대할 수 있다. 의료 분류에서는 정확도뿐 아니라 실제 악성을 놓치지 않는 정도인 악성 재현율도 함께 확인해야 한다.

## 7. 결론

세포핵의 평균 반지름, 오목함, 대칭성만으로도 양성과 악성을 상당 부분 구분할 수 있다. 그러나 세 개의 변수만 사용하므로 전체 30개 특성을 활용한 모델보다 분류 성능이 낮을 수 있다. 따라서 이 결과는 실제 의료 진단을 대신하는 것이 아니라, 핵심적인 세포핵 특성과 분류 결과의 관계를 학습하기 위한 데이터 분석 결과로 해석해야 한다.

## 8. 파일 구성

```text
breast_cancer_3features/
├── main.py
├── app.py
├── README.md
├── requirements.txt
└── results/              # main.py 실행 후 자동 생성
```
