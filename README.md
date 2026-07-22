서울 따릉이 생활지수 Streamlit 앱
파일 구성
`app.py`: Streamlit 웹앱 코드
`requirements.txt`: 배포에 필요한 패키지
`data.xlsx`: 서울시 공공자전거 자치구별 대여건수 원본
로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```
Streamlit Community Cloud 배포
GitHub에 새 저장소를 만듭니다.
이 폴더의 세 파일을 저장소 최상위 경로에 업로드합니다.
Streamlit Community Cloud에서 Create app을 선택합니다.
Repository와 branch를 선택하고 Main file path에 `app.py`를 입력합니다.
Deploy를 누릅니다.
생활지수 산식
이용규모 45%
연중 지속성 25%
성수기 활력 15%
2020년 대비 성장성 15%
각 항목은 25개 자치구 내 백분위 점수로 변환해 합산합니다.
유사 자치구 산식
12개월별 이용 비중
연간 이용 규모
월별 변동성
2020→2021 성장률
위 변수를 표준화한 뒤 유클리드 거리로 가장 가까운 자치구를 찾습니다.
