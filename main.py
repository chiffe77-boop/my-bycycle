from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="서울 따릉이 생활지수",
    page_icon="🚲",
    layout="wide",
)

DATA_FILE = Path(__file__).parent / "서울시 공공자전거 자치구별 대여건수(2021년).xlsx"
GEOJSON_URL = (
    "https://raw.githubusercontent.com/southkorea/seoul-maps/"
    "master/kostat/2013/json/seoul_municipalities_geo_simple.json"
)
TARGET_YEAR = 2021
MONTH_LABELS = [f"{m}월" for m in range(1, 13)]
SEASONS = {
    "겨울": [12, 1, 2],
    "봄": [3, 4, 5],
    "여름": [6, 7, 8],
    "가을": [9, 10, 11],
}

# -------------------------
# Design helpers
# -------------------------
st.markdown(
    """
    <style>
    .main-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.1rem;}
    .sub-title {color: #5f6b7a; margin-bottom: 1.2rem;}
    .insight-card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin: .35rem 0;
        background: rgba(250,250,250,.04);
    }
    .score-box {
        border-radius: 18px;
        padding: 1.2rem;
        border: 1px solid rgba(128,128,128,.22);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def pct_rank(series: pd.Series) -> pd.Series:
    """Convert a metric to a 0-100 percentile score."""
    return series.rank(pct=True, method="average") * 100


def life_grade(score: float) -> tuple[str, str]:
    """생활지수를 기억하기 쉬운 등급과 유형으로 변환한다."""
    if score >= 90:
        return "A+", "생활 정착형"
    if score >= 80:
        return "A", "생활 활성형"
    if score >= 70:
        return "B", "성장 생활형"
    if score >= 60:
        return "C", "생활 잠재형"
    return "D", "생활 기반 형성형"


@st.cache_data(show_spinner=False)
def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path.name}")

    df = pd.read_excel(path, sheet_name="자치구별 대여건수")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")

    # 숫자 월과 '소계'가 혼재하므로 정리
    df["년"] = pd.to_numeric(df["년"], errors="coerce")
    df = df[df["년"].notna()].copy()
    df["년"] = df["년"].astype(int)

    district_cols = [
        c for c in df.columns
        if c not in ["년", "월", "계", "기타"] and str(c).endswith("구")
    ]

    for col in ["계", *district_cols]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["년", "월", "계", *district_cols]]


@st.cache_data(show_spinner=False)
def prepare_analysis(df: pd.DataFrame, year: int = TARGET_YEAR):
    district_cols = [c for c in df.columns if c.endswith("구")]

    annual_row = df[(df["년"] == year) & (df["월"].astype(str) == "소계")]
    if annual_row.empty:
        raise ValueError(f"{year}년 소계 행이 없습니다.")

    monthly = df[(df["년"] == year) & (df["월"].astype(str) != "소계")].copy()
    monthly["월"] = pd.to_numeric(monthly["월"], errors="coerce")
    monthly = monthly[monthly["월"].between(1, 12)].sort_values("월")

    if len(monthly) != 12:
        raise ValueError(f"{year}년 월별 데이터가 12개월 완전하지 않습니다.")

    annual = annual_row.iloc[0][district_cols].astype(float)
    month_matrix = monthly.set_index("월")[district_cols].T
    month_matrix.columns = [int(c) for c in month_matrix.columns]
    month_share = month_matrix.div(month_matrix.sum(axis=1), axis=0)

    # 전년 성장률
    prev_row = df[(df["년"] == year - 1) & (df["월"].astype(str) == "소계")]
    if not prev_row.empty:
        prev_annual = prev_row.iloc[0][district_cols].astype(float)
        growth = (annual / prev_annual - 1) * 100
    else:
        growth = pd.Series(np.nan, index=district_cols)

    # 생활지수의 구성요소
    # 1) 이용 활성도: 연간 대여량이 많은 정도
    volume_score = pct_rank(annual)

    # 2) 연중 지속성: 가장 한산한 달에도 이용이 유지되는 정도
    monthly_mean = month_matrix.mean(axis=1)
    low_season_retention = month_matrix.min(axis=1) / monthly_mean
    consistency_score = pct_rank(low_season_retention)

    # 3) 성장성: 전년 대비 이용 증가율
    growth_score = pct_rank(growth.fillna(growth.median()))

    # 4) 계절 안정성: 월별 변동폭이 작은 정도
    cv = month_matrix.std(axis=1) / monthly_mean
    stability_score = pct_rank(-cv)

    # 서울 따릉이 생활지수
    # 이용 활성도 40% + 연중 지속성 30% + 성장성 20% + 계절 안정성 10%
    life_index = (
        volume_score * 0.40
        + consistency_score * 0.30
        + growth_score * 0.20
        + stability_score * 0.10
    )

    warm_share = month_share[[4, 5, 6, 7, 8, 9, 10]].sum(axis=1) * 100
    peak_month = month_matrix.idxmax(axis=1)
    low_month = month_matrix.idxmin(axis=1)

    summary = pd.DataFrame({
        "자치구": district_cols,
        "연간 대여건수": annual.values,
        "전년 대비 성장률": growth.reindex(district_cols).values,
        "월 변동계수": cv.reindex(district_cols).values,
        "비수기 유지율": low_season_retention.reindex(district_cols).values,
        "따뜻한 계절 비중": warm_share.reindex(district_cols).values,
        "최고 이용월": peak_month.reindex(district_cols).values,
        "최저 이용월": low_month.reindex(district_cols).values,
        "이용규모 점수": volume_score.reindex(district_cols).values,
        "연중 지속성 점수": consistency_score.reindex(district_cols).values,
        "성장성 점수": growth_score.reindex(district_cols).values,
        "계절 안정성 점수": stability_score.reindex(district_cols).values,
        "서울 따릉이 생활지수": life_index.reindex(district_cols).values,
    }).sort_values("서울 따릉이 생활지수", ascending=False)

    summary["생활지수 순위"] = np.arange(1, len(summary) + 1)
    grades = summary["서울 따릉이 생활지수"].apply(life_grade)
    summary["생활지수 등급"] = grades.str[0]
    summary["생활유형"] = grades.str[1]
    summary = summary.set_index("자치구", drop=False)

    # 유사도: 월별 이용구조(12개) + 규모 + 변동성 + 성장률
    similarity_features = month_share.copy()
    similarity_features.columns = [f"month_share_{m}" for m in similarity_features.columns]
    similarity_features["log_volume"] = np.log1p(annual)
    similarity_features["seasonality_cv"] = cv
    similarity_features["growth"] = growth.fillna(growth.median())

    scaler = StandardScaler()
    scaled = scaler.fit_transform(similarity_features)
    scaled_df = pd.DataFrame(scaled, index=similarity_features.index)

    return summary, month_matrix, month_share, scaled_df


@st.cache_data(ttl=86400, show_spinner=False)
def load_geojson():
    try:
        response = requests.get(GEOJSON_URL, timeout=8)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def similarity_table(selected: str, scaled_df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    base = scaled_df.loc[selected].values
    distances = np.sqrt(((scaled_df - base) ** 2).sum(axis=1))
    distances = distances.drop(selected).sort_values()

    # 가장 가까운 자치구를 100점에 가깝게 표현하는 직관적 점수
    max_dist = distances.max()
    sim_score = 100 * (1 - distances / max_dist) if max_dist > 0 else 100

    result = pd.DataFrame({
        "자치구": distances.index,
        "유사도 점수": sim_score.values,
        "생활지수": summary.loc[distances.index, "서울 따릉이 생활지수"].values,
        "연간 대여건수": summary.loc[distances.index, "연간 대여건수"].values,
        "성장률": summary.loc[distances.index, "전년 대비 성장률"].values,
    })
    return result.sort_values("유사도 점수", ascending=False)


def district_insights(selected: str, summary: pd.DataFrame, month_matrix: pd.DataFrame) -> list[str]:
    row = summary.loc[selected]
    rank = int(row["생활지수 순위"])
    total = int(row["연간 대여건수"])
    growth = row["전년 대비 성장률"]
    peak_m = int(row["최고 이용월"])
    low_m = int(row["최저 이용월"])
    peak_value = int(month_matrix.loc[selected, peak_m])
    avg_value = month_matrix.loc[selected].mean()

    insights = [
        f"**{selected}의 생활지수는 {row['서울 따릉이 생활지수']:.1f}점({row['생활지수 등급']}, {row['생활유형']})으로 서울 25개 자치구 중 {rank}위**입니다.",
        f"2021년 연간 대여건수는 **{total:,.0f}건**이며, 가장 많이 이용한 달은 **{peak_m}월({peak_value:,.0f}건)**입니다.",
        f"최고 이용월은 월평균보다 **{(peak_value / avg_value - 1) * 100:.1f}%** 높아 계절적 피크가 나타납니다.",
    ]

    if pd.notna(growth):
        direction = "증가" if growth >= 0 else "감소"
        insights.append(f"2020년 대비 이용량은 **{abs(growth):.1f}% {direction}**했습니다.")

    if row["연중 지속성 점수"] >= 70:
        insights.append("월별 편차가 비교적 작아 **일상형·상시형 이용 패턴**에 가깝습니다.")
    elif row["연중 지속성 점수"] <= 30:
        insights.append("월별 편차가 큰 편이어서 **계절·날씨 영향을 강하게 받는 이용 패턴**입니다.")

    insights.append(f"이용이 가장 적은 달은 **{low_m}월**로, 비수기 활성화가 정책·마케팅 기회가 될 수 있습니다.")
    return insights




def build_clusters(month_share: pd.DataFrame, summary: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """월별 이용구조와 규모·지속성·성장성을 바탕으로 자치구 이용 유형을 분류한다."""
    features = month_share.copy()
    features["log_volume"] = np.log1p(summary.loc[features.index, "연간 대여건수"])
    features["consistency"] = summary.loc[features.index, "연중 지속성 점수"] / 100
    features["growth"] = summary.loc[features.index, "전년 대비 성장률"].fillna(0) / 100

    # 월 컬럼은 정수(1~12), 추가 지표 컬럼은 문자열이므로
    # scikit-learn 1.2+에서 혼합 컬럼명 오류가 발생할 수 있다.
    features.columns = features.columns.astype(str)
    scaled = StandardScaler().fit_transform(features)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = model.fit_predict(scaled)

    result = pd.DataFrame(index=features.index)
    result["cluster"] = labels
    cluster_stats = []
    for cluster_id in sorted(result["cluster"].unique()):
        members = result.index[result["cluster"] == cluster_id]
        monthly_avg = month_share.loc[members].mean()
        warm = monthly_avg[[4, 5, 6, 7, 8, 9, 10]].sum()
        winter = monthly_avg[[12, 1, 2]].sum()
        vol_pct = summary.loc[members, "연간 대여건수"].rank(pct=True).mean()
        consistency = summary.loc[members, "연중 지속성 점수"].mean()
        growth = summary.loc[members, "전년 대비 성장률"].mean()
        peak = int(monthly_avg.idxmax())
        if consistency >= 60 and winter >= 0.17:
            label = "연중 생활형"
        elif warm >= 0.66:
            label = "봄·여름 레저형"
        elif growth >= summary["전년 대비 성장률"].median():
            label = "성장 가속형"
        elif vol_pct >= 0.55:
            label = "대규모 수요형"
        else:
            label = "계절 민감형"
        cluster_stats.append((cluster_id, label, peak))

    label_map = {c: label for c, label, _ in cluster_stats}
    peak_map = {c: peak for c, _, peak in cluster_stats}
    result["이용 유형"] = result["cluster"].map(label_map)
    result["대표 피크월"] = result["cluster"].map(peak_map)
    return result


def hidden_champions(summary: pd.DataFrame) -> pd.DataFrame:
    """지속성·성장성·계절 안정성으로 기대 이용량을 만들고 실제 이용량과의 차이를 계산한다."""
    work = summary.copy()
    X = work[["연중 지속성 점수", "계절 안정성 점수", "성장성 점수"]].fillna(50)
    y = np.log1p(work["연간 대여건수"])
    model = LinearRegression().fit(X, y)
    expected = np.expm1(model.predict(X))
    work["기대 대여건수"] = expected
    work["기대 대비 초과율"] = (work["연간 대여건수"] / work["기대 대여건수"] - 1) * 100
    work["숨은 강자 점수"] = pct_rank(work["기대 대비 초과율"])
    return work.sort_values("기대 대비 초과율", ascending=False)


def policy_recommendations(selected: str, summary: pd.DataFrame) -> list[str]:
    row = summary.loc[selected]
    recs = []
    if row["연중 지속성 점수"] >= 70:
        recs.append("출퇴근·생활권 중심의 상시 거치소 운영과 정기권 커뮤니케이션을 강화합니다.")
    else:
        recs.append("비수기 이용 촉진을 위해 계절별 쿠폰·안전 캠페인·관광 동선 연계를 검토합니다.")
    if row["이용규모 점수"] >= 70:
        recs.append("수요가 큰 지역이므로 피크 시간대 재배치와 대여소 포화 모니터링이 우선입니다.")
    else:
        recs.append("절대 이용량 확대보다 잠재 수요가 있는 생활권과 환승 거점의 선택적 확장이 적절합니다.")
    if row["성장성 점수"] >= 70:
        recs.append("성장세가 빠르므로 신규 거치소 후보지와 자전거도로 연결 구간을 선제적으로 점검합니다.")
    elif row["성장성 점수"] <= 30:
        recs.append("성장 둔화 원인을 대여소 접근성·차량 가용성·경쟁 교통수단 관점에서 진단합니다.")
    if row["계절 안정성 점수"] >= 70:
        recs.append("계절 변화에도 이용이 안정적이므로 출퇴근·생활권 중심의 상시 운영과 정기 이용 프로그램이 적합합니다.")
    elif row["계절 안정성 점수"] <= 30:
        recs.append("계절 변동이 커 비수기 프로모션과 날씨 대응형 운영으로 연중 이용 기반을 넓힐 필요가 있습니다.")
    return recs[:4]


def city_story(summary: pd.DataFrame, month_matrix: pd.DataFrame) -> str:
    top = summary.nlargest(1, "서울 따릉이 생활지수").iloc[0]
    growth = summary.nlargest(1, "전년 대비 성장률").iloc[0]
    stable = summary.nlargest(1, "연중 지속성 점수").iloc[0]
    peak_month = int(month_matrix.sum(axis=0).idxmax())
    return (
        f"2021년 서울 따릉이의 중심은 **{top['자치구']}**였습니다. "
        f"서울 전체 이용은 **{peak_month}월**에 정점을 찍었고, "
        f"가장 빠르게 성장한 곳은 **{growth['자치구']}**, "
        f"계절 변화에도 가장 꾸준했던 곳은 **{stable['자치구']}**입니다."
    )

# -------------------------
# Load and validate
# -------------------------
try:
    raw = load_raw_data(DATA_FILE)
    summary, month_matrix, month_share, scaled_df = prepare_analysis(raw)
    cluster_df = build_clusters(month_share, summary)
    champion_df = hidden_champions(summary)
    summary["이용 유형"] = cluster_df["이용 유형"]
except Exception as exc:
    st.error(f"데이터를 불러오지 못했습니다: {exc}")
    st.stop()

# -------------------------
# Header + sidebar
# -------------------------
st.markdown('<div class="main-title">🚲 서울 따릉이 생활지수: 따릉이는 우리 동네의 일상이 되었을까?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">따릉이가 서울 시민의 일상 교통수단으로 얼마나 자리 잡았는지 자치구별로 비교하는 데이터 인사이트 앱</div>',
    unsafe_allow_html=True,
)
st.info(
    "**서울 따릉이 생활지수**는 단순히 많이 탄 지역을 보여주는 순위가 아닙니다. "
    "따릉이가 한 지역에서 **얼마나 활발하고, 일 년 내내 꾸준하며, 성장하고, 계절 변화에도 안정적으로 이용되는지**를 종합한 상대평가 지수입니다."
)

with st.sidebar:
    st.header("분석 설정")
    selected_gu = st.selectbox(
        "자치구 선택",
        summary.sort_values("생활지수 순위")["자치구"].tolist(),
        index=summary.sort_values("생활지수 순위")["자치구"].tolist().index("마포구"),
    )
    top_n = st.slider("순위 표시 개수", 5, 25, 10)
    st.divider()
    st.caption("생활지수 산식")
    st.markdown(
        "- 이용 활성도 40%\n"
        "- 연중 지속성 30%\n"
        "- 성장성 20%\n"
        "- 계절 안정성 10%"
    )
    st.info(
        "이 지수는 공공자전거 이용 데이터만으로 만든 상대평가 지수입니다. "
        "인구·면적·대여소 수를 반영한 정책지수는 추가 데이터 결합이 필요합니다."
    )

# -------------------------
# KPI row
# -------------------------
seoul_total = int(summary["연간 대여건수"].sum())
leader = summary.sort_values("생활지수 순위").iloc[0]
peak_all = month_matrix.sum(axis=0).idxmax()
selected_row = summary.loc[selected_gu]

k1, k2, k3, k4 = st.columns(4)
k1.metric("서울 전체 대여건수", f"{seoul_total/1_000_000:.1f}M건")
k2.metric("생활지수 1위", leader["자치구"], f"{leader['서울 따릉이 생활지수']:.1f}점")
k3.metric("서울 최고 이용월", f"{int(peak_all)}월", f"{month_matrix.sum(axis=0).loc[peak_all]/1_000_000:.2f}M건")
k4.metric(
    f"{selected_gu} 생활지수",
    f"{selected_row['서울 따릉이 생활지수']:.1f}점 · {selected_row['생활지수 등급']}",
    f"서울 {int(selected_row['생활지수 순위'])}위 · {selected_row['생활유형']}",
)

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "서울 한눈에", "생활지수", "월별 패턴", "비슷한 자치구",
    "이용 유형", "숨은 강자", "성장 시뮬레이션", "자동 인사이트"
])

with tab1:
    st.success(city_story(summary, month_matrix))
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("자치구별 연간 대여건수")
        geojson = load_geojson()
        map_df = summary.reset_index(drop=True)
        if geojson:
            fig_map = px.choropleth_mapbox(
                map_df,
                geojson=geojson,
                locations="자치구",
                featureidkey="properties.name",
                color="연간 대여건수",
                color_continuous_scale="Blues",
                mapbox_style="carto-positron",
                zoom=9.4,
                center={"lat": 37.5665, "lon": 126.9780},
                opacity=0.72,
                hover_name="자치구",
                hover_data={
                    "연간 대여건수": ":,.0f",
                    "서울 따릉이 생활지수": ":.1f",
                    "생활지수 순위": True,
                },
            )
            fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=520)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("지도 경계 데이터를 불러오지 못해 순위 그래프로 대체했습니다.")
            fallback = map_df.nlargest(top_n, "연간 대여건수").sort_values("연간 대여건수")
            fig_fallback = px.bar(
                fallback,
                x="연간 대여건수",
                y="자치구",
                orientation="h",
                text_auto=".3s",
            )
            fig_fallback.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_fallback, use_container_width=True)

    with right:
        st.subheader(f"TOP {top_n} 이용 자치구")
        rank_df = summary.nlargest(top_n, "연간 대여건수").sort_values("연간 대여건수")
        fig_rank = px.bar(
            rank_df,
            x="연간 대여건수",
            y="자치구",
            orientation="h",
            text_auto=".3s",
            hover_data={"서울 따릉이 생활지수": ":.1f"},
        )
        fig_rank.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_rank, use_container_width=True)

    st.subheader("서울 전체 월별 이용 흐름")
    city_month = month_matrix.sum(axis=0).reset_index()
    city_month.columns = ["월", "대여건수"]
    city_month["월 라벨"] = city_month["월"].astype(int).astype(str) + "월"
    fig_city = px.line(city_month, x="월 라벨", y="대여건수", markers=True)
    fig_city.update_traces(line_width=3, marker_size=8)
    fig_city.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_city, use_container_width=True)

with tab2:
    st.subheader("서울 따릉이 생활지수 순위")
    rank_index = summary.nsmallest(top_n, "생활지수 순위").sort_values("서울 따릉이 생활지수")
    fig_index = px.bar(
        rank_index,
        x="서울 따릉이 생활지수",
        y="자치구",
        orientation="h",
        text="서울 따릉이 생활지수",
        hover_data={
            "이용규모 점수": ":.1f",
            "연중 지속성 점수": ":.1f",
            "계절 안정성 점수": ":.1f",
            "성장성 점수": ":.1f",
        },
    )
    fig_index.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_index.update_xaxes(range=[0, 105])
    fig_index.update_layout(height=520, margin=dict(l=0, r=20, t=10, b=0))
    st.plotly_chart(fig_index, use_container_width=True)

    st.subheader(f"{selected_gu} 지수 구성")
    radar_categories = ["이용 활성도", "연중 지속성", "성장성", "계절 안정성"]
    radar_values = [
        selected_row["이용규모 점수"],
        selected_row["연중 지속성 점수"],
        selected_row["성장성 점수"],
        selected_row["계절 안정성 점수"],
    ]
    radar_values_closed = radar_values + [radar_values[0]]
    radar_categories_closed = radar_categories + [radar_categories[0]]
    fig_radar = go.Figure(
        data=go.Scatterpolar(
            r=radar_values_closed,
            theta=radar_categories_closed,
            fill="toself",
            name=selected_gu,
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=430,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    st.subheader(f"{selected_gu} 월별 이용 패턴")
    district_month = month_matrix.loc[selected_gu].reset_index()
    district_month.columns = ["월", "대여건수"]
    district_month["월 라벨"] = district_month["월"].astype(int).astype(str) + "월"
    city_avg = month_matrix.mean(axis=0).reset_index()
    city_avg.columns = ["월", "서울 자치구 평균"]
    pattern = district_month.merge(city_avg, on="월")

    fig_pattern = go.Figure()
    fig_pattern.add_trace(go.Scatter(
        x=pattern["월 라벨"], y=pattern["대여건수"],
        mode="lines+markers", name=selected_gu,
        line=dict(width=4), marker=dict(size=8),
    ))
    fig_pattern.add_trace(go.Scatter(
        x=pattern["월 라벨"], y=pattern["서울 자치구 평균"],
        mode="lines+markers", name="서울 자치구 평균",
        line=dict(width=2, dash="dash"), marker=dict(size=6),
    ))
    fig_pattern.update_layout(height=430, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_pattern, use_container_width=True)

    season_rows = []
    for season, months in SEASONS.items():
        season_rows.append({
            "계절": season,
            "대여건수": month_matrix.loc[selected_gu, months].sum(),
        })
    season_df = pd.DataFrame(season_rows)
    fig_season = px.pie(
        season_df,
        names="계절",
        values="대여건수",
        hole=0.45,
        title=f"{selected_gu} 계절별 이용 비중",
    )
    fig_season.update_layout(height=400, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_season, use_container_width=True)

with tab4:
    st.subheader(f"{selected_gu}와 가장 비슷한 자치구")
    similar = similarity_table(selected_gu, scaled_df, summary)
    top_similar = similar.head(5).copy()

    fig_sim = px.bar(
        top_similar.sort_values("유사도 점수"),
        x="유사도 점수",
        y="자치구",
        orientation="h",
        text="유사도 점수",
        hover_data={"생활지수": ":.1f", "연간 대여건수": ":,.0f", "성장률": ":.1f"},
    )
    fig_sim.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_sim.update_xaxes(range=[0, 105])
    fig_sim.update_layout(height=380, margin=dict(l=0, r=20, t=10, b=0))
    st.plotly_chart(fig_sim, use_container_width=True)

    compare_gus = [selected_gu] + top_similar["자치구"].head(3).tolist()
    compare_df = month_share.loc[compare_gus].T.reset_index()
    compare_df.columns = ["월", *compare_gus]
    compare_long = compare_df.melt(id_vars="월", var_name="자치구", value_name="월별 비중")
    compare_long["월 라벨"] = compare_long["월"].astype(int).astype(str) + "월"
    compare_long["월별 비중"] *= 100

    fig_compare = px.line(
        compare_long,
        x="월 라벨",
        y="월별 비중",
        color="자치구",
        markers=True,
        title="연간 이용량 중 월별 비중 비교",
    )
    fig_compare.update_layout(height=430, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_compare, use_container_width=True)

    st.caption(
        "유사도는 12개월 이용 비중, 연간 규모, 월별 변동성, 2020→2021 성장률을 표준화한 뒤 거리로 계산합니다."
    )
    st.dataframe(
        top_similar.style.format({
            "유사도 점수": "{:.1f}",
            "생활지수": "{:.1f}",
            "연간 대여건수": "{:,.0f}",
            "성장률": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )


with tab5:
    st.subheader("서울 따릉이 이용 유형 지도")
    st.caption("월별 이용 비중, 이용 규모, 지속성, 성장성을 함께 반영한 비지도학습 군집입니다.")
    cluster_plot = summary.reset_index(drop=True).copy()
    fig_cluster = px.scatter(
        cluster_plot,
        x="연중 지속성 점수",
        y="전년 대비 성장률",
        size="연간 대여건수",
        color="이용 유형",
        hover_name="자치구",
        hover_data={"서울 따릉이 생활지수": ":.1f", "연간 대여건수": ":,.0f"},
        size_max=46,
    )
    fig_cluster.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_cluster, use_container_width=True)

    selected_type = summary.loc[selected_gu, "이용 유형"]
    peers = summary[summary["이용 유형"] == selected_type].sort_values("서울 따릉이 생활지수", ascending=False)
    st.info(f"**{selected_gu}는 ‘{selected_type}’ 유형**입니다. 같은 유형: {', '.join(peers['자치구'].tolist())}")
    st.dataframe(
        peers[["자치구", "이용 유형", "서울 따릉이 생활지수", "연간 대여건수", "전년 대비 성장률"]]
        .style.format({"서울 따릉이 생활지수": "{:.1f}", "연간 대여건수": "{:,.0f}", "전년 대비 성장률": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )

with tab6:
    st.subheader("규모만 봐서는 놓치는 ‘숨은 강자’")
    st.caption("연중 지속성·계절 안정성·성장성으로 기대 이용량을 추정한 뒤, 실제 이용량이 기대보다 얼마나 높은지 비교합니다.")
    top_champ = champion_df.head(top_n).sort_values("기대 대비 초과율")
    fig_champ = px.bar(
        top_champ,
        x="기대 대비 초과율",
        y="자치구",
        orientation="h",
        text="기대 대비 초과율",
        hover_data={"연간 대여건수": ":,.0f", "기대 대여건수": ":,.0f", "숨은 강자 점수": ":.1f"},
    )
    fig_champ.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_champ.update_layout(height=500, margin=dict(l=0, r=30, t=10, b=0))
    st.plotly_chart(fig_champ, use_container_width=True)

    champ_row = champion_df.loc[selected_gu]
    gap = champ_row["기대 대비 초과율"]
    if gap >= 0:
        st.success(f"{selected_gu}는 분석모형의 기대치보다 실제 이용량이 **{gap:.1f}% 높아**, 잠재력이 실제 수요로 잘 전환된 지역입니다.")
    else:
        st.warning(f"{selected_gu}는 분석모형의 기대치보다 실제 이용량이 **{abs(gap):.1f}% 낮아**, 접근성·대여소 배치·인지도 개선 여지가 있습니다.")

with tab7:
    st.subheader(f"{selected_gu} 성장 시뮬레이션")
    growth_assumption = st.slider("연간 대여건수 변화 가정", -30, 50, 10, 5, format="%d%%")
    scenario = summary.copy()
    base_volume = scenario.loc[selected_gu, "연간 대여건수"]
    scenario.loc[selected_gu, "연간 대여건수"] = base_volume * (1 + growth_assumption / 100)
    scenario["시나리오 규모점수"] = pct_rank(scenario["연간 대여건수"])
    scenario["시나리오 생활지수"] = (
        scenario["시나리오 규모점수"] * 0.40
        + scenario["연중 지속성 점수"] * 0.30
        + scenario["성장성 점수"] * 0.20
        + scenario["계절 안정성 점수"] * 0.10
    )
    scenario = scenario.sort_values("시나리오 생활지수", ascending=False)
    scenario["시나리오 순위"] = np.arange(1, len(scenario) + 1)
    old_rank = int(summary.loc[selected_gu, "생활지수 순위"])
    new_rank = int(scenario.loc[selected_gu, "시나리오 순위"])
    old_score = summary.loc[selected_gu, "서울 따릉이 생활지수"]
    new_score = scenario.loc[selected_gu, "시나리오 생활지수"]
    city_effect = (scenario["연간 대여건수"].sum() / summary["연간 대여건수"].sum() - 1) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재 순위", f"{old_rank}위")
    c2.metric("시나리오 순위", f"{new_rank}위", f"{old_rank-new_rank:+d}계단")
    c3.metric("생활지수 변화", f"{new_score:.1f}점", f"{new_score-old_score:+.1f}점")
    c4.metric("서울 전체 이용 변화", f"{city_effect:+.2f}%")

    compare_scenario = pd.DataFrame({
        "구분": ["현재", "시나리오"],
        "연간 대여건수": [base_volume, scenario.loc[selected_gu, "연간 대여건수"]],
        "생활지수": [old_score, new_score],
    })
    fig_scenario = px.bar(compare_scenario, x="구분", y="연간 대여건수", text_auto=".3s", hover_data={"생활지수": ":.1f"})
    fig_scenario.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_scenario, use_container_width=True)


with tab8:
    st.subheader(f"{selected_gu} 자동 인사이트")
    for text in district_insights(selected_gu, summary, month_matrix):
        st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)

    st.subheader("데이터 기반 실행 제안")
    for i, rec in enumerate(policy_recommendations(selected_gu, summary), 1):
        st.markdown(f'<div class="insight-card"><b>{i}.</b> {rec}</div>', unsafe_allow_html=True)

    st.subheader("서울 전체에서 발견한 핵심 신호")
    top_volume = summary.nlargest(1, "연간 대여건수").iloc[0]
    top_growth = summary.nlargest(1, "전년 대비 성장률").iloc[0]
    top_consistency = summary.nlargest(1, "연중 지속성 점수").iloc[0]
    concentration_top5 = summary.nlargest(5, "연간 대여건수")["연간 대여건수"].sum() / summary["연간 대여건수"].sum() * 100

    city_insights = [
        f"2021년 서울 전체 따릉이 대여는 **{seoul_total:,.0f}건**입니다.",
        f"대여건수 1위는 **{top_volume['자치구']}({top_volume['연간 대여건수']:,.0f}건)**입니다.",
        f"2020년 대비 성장률이 가장 높은 곳은 **{top_growth['자치구']}({top_growth['전년 대비 성장률']:.1f}%)**입니다.",
        f"월별 이용이 가장 안정적인 곳은 **{top_consistency['자치구']}**입니다.",
        f"상위 5개 자치구가 서울 전체 대여건수의 **{concentration_top5:.1f}%**를 차지합니다.",
    ]
    for text in city_insights:
        st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)

    st.subheader("분석 데이터 내려받기")
    export_cols = [
        "자치구", "생활지수 순위", "서울 따릉이 생활지수", "연간 대여건수",
        "전년 대비 성장률", "이용규모 점수", "연중 지속성 점수",
        "계절 안정성 점수", "성장성 점수", "최고 이용월", "최저 이용월",
    ]
    csv_data = summary[export_cols].sort_values("생활지수 순위").to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "생활지수 분석결과 CSV 다운로드",
        data=csv_data,
        file_name="seoul_bike_life_index_2021.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "데이터: 서울시 공공자전거 자치구별 대여건수. "
    "생활지수는 본 웹앱이 정의한 상대평가 지수이며 공식 서울시 지표가 아닙니다."
)
