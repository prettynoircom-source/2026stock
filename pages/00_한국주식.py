import subprocess
import sys
import importlib

# ──────────────────────────────────────────────
# 필수 패키지 자동 설치 (requirements.txt 없이 이 파일 하나만으로 실행 가능하게)
# ──────────────────────────────────────────────
_REQUIRED_PACKAGES = {
    "streamlit": "streamlit",
    "yfinance": "yfinance",
    "pandas": "pandas",
    "numpy": "numpy",
    "plotly": "plotly",
}

for _import_name, _pip_name in _REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(_import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pip_name])

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="한국 AI·반도체 대표주 분석",
    page_icon="🇰🇷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main .block-container {padding-top: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🇰🇷 한국 AI·반도체 대표주 분석 대시보드")
st.caption("삼성전자, SK하이닉스 등 국내 AI·반도체 핵심 종목의 인터랙티브 분석 도구 (Yahoo Finance 데이터 기반)")

# ──────────────────────────────────────────────
# 한국 AI·반도체 대표 종목 리스트 (표시명 → 티커)
# ──────────────────────────────────────────────
KOREA_AI_SEMI_STOCKS = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "삼성전자우": "005935.KS",
    "한미반도체": "042700.KS",
    "DB하이텍": "000990.KS",
    "리노공업": "058470.KQ",
    "이수페타시스": "007660.KS",
    "솔브레인": "357780.KQ",
    "원익IPS": "240810.KQ",
    "티씨케이": "064760.KQ",
    "코미코": "183300.KQ",
    "넥스틴": "348210.KQ",
    "네패스": "033640.KQ",
    "LG이노텍": "011070.KS",
    "삼성전기": "009150.KS",
    "SK스퀘어": "402340.KS",
    "네이버": "035420.KS",
    "카카오": "035720.KS",
    "더존비즈온": "012510.KS",
    "루닛": "328130.KQ",
}

DEFAULT_SELECTION = ["삼성전자", "SK하이닉스", "한미반도체", "네이버"]

# ──────────────────────────────────────────────
# 데이터 로딩 함수 (캐싱)
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


# ──────────────────────────────────────────────
# 기술적 지표 계산 함수
# ──────────────────────────────────────────────
def add_moving_averages(df, windows=(20, 50, 200)):
    for w in windows:
        df[f"SMA{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_bollinger_bands(df, window=20, num_std=2):
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    df["BB_MID"] = sma
    df["BB_UPPER"] = sma + num_std * std
    df["BB_LOWER"] = sma - num_std * std
    return df


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def fmt_krw(value):
    """원화 포맷"""
    try:
        return f"₩{value:,.0f}"
    except (TypeError, ValueError):
        return "N/A"


# ──────────────────────────────────────────────
# 사이드바 - 사용자 입력
# ──────────────────────────────────────────────
st.sidebar.header("⚙️ 설정")

selected_names = st.sidebar.multiselect(
    "종목 선택 (AI·반도체 대표주)",
    options=list(KOREA_AI_SEMI_STOCKS.keys()),
    default=DEFAULT_SELECTION,
    help="여러 종목을 선택하면 비교 분석도 가능합니다."
)

with st.sidebar.expander("➕ 직접 티커 추가 (선택)"):
    custom_ticker_input = st.text_input(
        "쉼표로 구분하여 입력",
        value="",
        help="예: 000670.KS (목록에 없는 국내 종목 추가 시, 코드 뒤에 .KS 또는 .KQ 붙이기)"
    )

custom_tickers = [t.strip().upper() for t in custom_ticker_input.split(",") if t.strip()]
selected_tickers = [KOREA_AI_SEMI_STOCKS[name] for name in selected_names] + custom_tickers
ticker_to_name = {v: k for k, v in KOREA_AI_SEMI_STOCKS.items()}

col_a, col_b = st.sidebar.columns(2)
with col_a:
    start_date = st.date_input("시작일", value=datetime.today() - timedelta(days=365))
with col_b:
    end_date = st.date_input("종료일", value=datetime.today())

st.sidebar.markdown("---")
st.sidebar.subheader("차트 옵션")
chart_type = st.sidebar.radio("메인 차트 유형", ["캔들스틱", "라인"], horizontal=True)

show_sma = st.sidebar.checkbox("이동평균선 (SMA 20/50/200)", value=True)
show_bb = st.sidebar.checkbox("볼린저 밴드", value=False)
show_volume = st.sidebar.checkbox("거래량 표시", value=True)

st.sidebar.markdown("---")

if not selected_tickers:
    st.warning("사이드바에서 최소 하나의 종목을 선택해주세요.")
    st.stop()

display_labels = [ticker_to_name.get(t, t) for t in selected_tickers]
primary_label = st.sidebar.selectbox("상세 분석 종목 선택", display_labels)
primary_ticker = selected_tickers[display_labels.index(primary_label)]

if start_date >= end_date:
    st.error("시작일은 종료일보다 이전이어야 합니다.")
    st.stop()

# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────
with st.spinner("데이터를 불러오는 중..."):
    data_dict = {}
    failed = []
    for t in selected_tickers:
        df = load_data(t, str(start_date), str(end_date))
        if not df.empty:
            data_dict[t] = df
        else:
            failed.append(t)

if failed:
    st.warning(f"다음 종목은 데이터를 불러오지 못했습니다: {', '.join(failed)}")

if primary_ticker not in data_dict:
    st.error(f"'{ticker_to_name.get(primary_ticker, primary_ticker)}' 데이터를 불러올 수 없습니다.")
    st.stop()

df_main = data_dict[primary_ticker].copy()
df_main = add_moving_averages(df_main)
df_main = add_bollinger_bands(df_main)
df_main["RSI"] = calculate_rsi(df_main["Close"])
macd_line, signal_line, hist = calculate_macd(df_main["Close"])
df_main["MACD"] = macd_line
df_main["MACD_SIGNAL"] = signal_line
df_main["MACD_HIST"] = hist

info = load_info(primary_ticker)
primary_display = ticker_to_name.get(primary_ticker, primary_ticker)

# ──────────────────────────────────────────────
# 상단 요약 지표
# ──────────────────────────────────────────────
last_close = df_main["Close"].iloc[-1]
prev_close = df_main["Close"].iloc[-2] if len(df_main) > 1 else last_close
change = last_close - prev_close
pct_change = (change / prev_close) * 100 if prev_close else 0
period_high = df_main["High"].max()
period_low = df_main["Low"].min()
avg_volume = df_main["Volume"].mean()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(f"{primary_display} 현재가", fmt_krw(last_close), f"{change:+,.0f} ({pct_change:+.2f}%)")
m2.metric("기간 최고가", fmt_krw(period_high))
m3.metric("기간 최저가", fmt_krw(period_low))
m4.metric("평균 거래량", f"{avg_volume:,.0f}")
if info.get("marketCap"):
    m5.metric("시가총액", f"₩{info['marketCap']/1e12:,.1f}조")
else:
    m5.metric("52주 최고", fmt_krw(info.get("fiftyTwoWeekHigh")) if info.get("fiftyTwoWeekHigh") else "N/A")

st.markdown("---")

# ──────────────────────────────────────────────
# 탭 구성
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 가격 차트", "🔍 기술적 지표", "⚖️ 종목 비교", "📋 데이터 테이블"])

# ── 탭 1: 가격 차트 ──────────────────────────────
with tab1:
    rows = 2 if show_volume else 1
    row_heights = [0.75, 0.25] if show_volume else [1.0]
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=row_heights
    )

    if chart_type == "캔들스틱":
        fig.add_trace(go.Candlestick(
            x=df_main.index, open=df_main["Open"], high=df_main["High"],
            low=df_main["Low"], close=df_main["Close"], name=primary_display,
            increasing_line_color="#d32f2f", decreasing_line_color="#1565c0"
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df_main.index, y=df_main["Close"], mode="lines",
            name=primary_display, line=dict(color="#2962ff", width=2)
        ), row=1, col=1)

    if show_sma:
        colors = {"SMA20": "#ff9800", "SMA50": "#9c27b0", "SMA200": "#00bcd4"}
        for sma_col, color in colors.items():
            if sma_col in df_main.columns:
                fig.add_trace(go.Scatter(
                    x=df_main.index, y=df_main[sma_col], mode="lines",
                    name=sma_col, line=dict(color=color, width=1.3)
                ), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main["BB_UPPER"], line=dict(color="rgba(150,150,150,0.5)", width=1), name="BB 상단"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_main.index, y=df_main["BB_LOWER"], line=dict(color="rgba(150,150,150,0.5)", width=1), name="BB 하단", fill="tonexty", fillcolor="rgba(150,150,150,0.1)"), row=1, col=1)

    if show_volume:
        vol_colors = np.where(df_main["Close"] >= df_main["Open"], "#d32f2f", "#1565c0")
        fig.add_trace(go.Bar(
            x=df_main.index, y=df_main["Volume"], name="거래량",
            marker_color=vol_colors, showlegend=False
        ), row=2, col=1)

    fig.update_layout(
        height=650, template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_yaxes(title_text="가격 (₩)", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="거래량", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

# ── 탭 2: 기술적 지표 ────────────────────────────
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RSI (상대강도지수)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_main.index, y=df_main["RSI"], line=dict(color="#673ab7", width=1.5), name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="과매수(70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="과매도(30)")
        fig_rsi.update_layout(height=350, template="plotly_white", yaxis_range=[0, 100], margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col2:
        st.subheader("MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df_main.index, y=df_main["MACD"], line=dict(color="#2962ff", width=1.5), name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df_main.index, y=df_main["MACD_SIGNAL"], line=dict(color="#ff6d00", width=1.5), name="Signal"))
        hist_colors = np.where(df_main["MACD_HIST"] >= 0, "#d32f2f", "#1565c0")
        fig_macd.add_trace(go.Bar(x=df_main.index, y=df_main["MACD_HIST"], marker_color=hist_colors, name="Histogram"))
        fig_macd.update_layout(height=350, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_macd, use_container_width=True)

    st.subheader("일별 수익률 분포")
    daily_returns = df_main["Close"].pct_change().dropna() * 100
    fig_hist = px.histogram(daily_returns, nbins=50, template="plotly_white",
                             labels={"value": "일별 수익률 (%)"}, title=None)
    fig_hist.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

# ── 탭 3: 종목 비교 ──────────────────────────────
with tab3:
    if len(data_dict) < 2:
        st.info("2개 이상의 종목을 선택하면 비교 차트가 표시됩니다.")
    else:
        st.subheader("정규화된 누적 수익률 비교 (시작일 = 100)")
        fig_compare = go.Figure()
        for t, df_t in data_dict.items():
            normalized = df_t["Close"] / df_t["Close"].iloc[0] * 100
            fig_compare.add_trace(go.Scatter(x=df_t.index, y=normalized, mode="lines", name=ticker_to_name.get(t, t)))
        fig_compare.update_layout(
            height=450, template="plotly_white", yaxis_title="정규화 지수 (기준=100)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        st.subheader("수익률 상관관계 히트맵")
        returns_df = pd.DataFrame({ticker_to_name.get(t, t): df_t["Close"].pct_change() for t, df_t in data_dict.items()}).dropna()
        corr = returns_df.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template="plotly_white")
        fig_corr.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("종목별 요약 통계")
        summary_rows = []
        for t, df_t in data_dict.items():
            ret = df_t["Close"].pct_change().dropna()
            total_return = (df_t["Close"].iloc[-1] / df_t["Close"].iloc[0] - 1) * 100
            summary_rows.append({
                "종목": ticker_to_name.get(t, t),
                "티커": t,
                "현재가(₩)": round(df_t["Close"].iloc[-1], 0),
                "기간 수익률(%)": round(total_return, 2),
                "연환산 변동성(%)": round(ret.std() * np.sqrt(252) * 100, 2),
                "최고가(₩)": round(df_t["High"].max(), 0),
                "최저가(₩)": round(df_t["Low"].min(), 0),
            })
        st.dataframe(pd.DataFrame(summary_rows).set_index("종목"), use_container_width=True)

# ── 탭 4: 데이터 테이블 ───────────────────────────
with tab4:
    st.subheader(f"{primary_display} 원본 데이터")
    display_df = df_main[["Open", "High", "Low", "Close", "Volume"]].sort_index(ascending=False)
    st.dataframe(display_df, use_container_width=True, height=450)

    csv = display_df.to_csv().encode("utf-8-sig")
    st.download_button(
        label="📥 CSV 다운로드",
        data=csv,
        file_name=f"{primary_display}_{start_date}_{end_date}.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("데이터 출처: Yahoo Finance (yfinance) · 대상: 국내 AI·반도체 대표 종목 · 투자 판단은 본인 책임이며 이 앱은 투자 자문을 제공하지 않습니다.")
