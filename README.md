# 🚲 서울시 공공자전거 자치구별 대여건수 대시보드 (2021년)

서울시 공공자전거(따릉이) 2021년 자치구별 월별 대여건수를 시각화하는 Streamlit 웹앱입니다.
자치구를 선택하면 위치 지도와 월별 대여 추이 그래프를 함께 확인할 수 있습니다.

## 🔗 데모

Streamlit Cloud에 배포 후 이 부분에 링크를 추가하세요.

```
https://<your-app-name>.streamlit.app
```

## ✨ 주요 기능

- **지역 선택**: 사이드바에서 자치구를 드롭다운으로 다중 선택하거나, 텍스트로 직접 입력(쉼표로 구분) 가능
- **위치 지도**: 선택한 자치구의 위치를 지도 위에 표시 (원 크기·색상 = 연간 대여건수)
- **월별 추이 그래프**: 선택한 자치구별 2021년 1~12월 대여건수 라인 그래프
- **연간 비교 막대그래프**: 선택 지역들의 연간 총 대여건수 비교
- **원본 데이터 확인**: 표 형태로 월별 데이터 확인 가능

## 🗂 프로젝트 구조

```
.
├── app.py              # Streamlit 앱 메인 코드 (데이터 내장)
├── requirements.txt    # 의존성 패키지 목록
└── README.md
```

데이터는 `app.py` 안에 직접 내장되어 있어 별도의 데이터 파일이 필요 없습니다.
(원본: 서울시 공공자전거 자치구별 대여건수_2021년, 서울 열린데이터광장)

## 🛠 기술 스택

- [Streamlit](https://streamlit.io/) — 웹앱 프레임워크
- [Plotly](https://plotly.com/python/) — 인터랙티브 시각화 (지도, 라인/막대 그래프)
- [Pandas](https://pandas.pydata.org/) — 데이터 처리

## 💻 로컬에서 실행하기

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속하면 앱을 확인할 수 있습니다.

## ☁️ Streamlit Cloud 배포하기

1. 이 저장소를 GitHub에 push
2. [share.streamlit.io](https://share.streamlit.io) 접속 후 로그인
3. **New app** 클릭 → 저장소, 브랜치 선택 → Main file path에 `app.py` 입력
4. **Deploy** 클릭

몇 분 내로 배포가 완료되고 공개 URL이 생성됩니다.

## 📊 데이터 출처

서울 열린데이터광장 — 서울시 공공자전거 대여건수 정보 (2021년, 자치구별)

## 📄 라이선스

원본 데이터의 이용 조건은 서울 열린데이터광장의 정책을 따릅니다.
