import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. 페이지 설정
st.set_page_config(
    page_title="기온 상승 트렌드 분석 앱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌡️ 1980년대 전후 기온 상승 트렌드 비교 분석")
st.markdown("""
제공된 과거 기온 데이터를 바탕으로 **1980년대 이전과 이후의 기온 상승 속도에 차이가 있는지** 분석하는 웹앱입니다.
지구 온난화 및 급격한 도시화가 진행된 1980년 전후의 데이터를 직접 비교해보세요.
""")

# 2. 데이터 로드 및 전처리 함수
@st.cache_data
def load_data(file_path):
    # CSV 파일 읽기 (날짜 공백 및 따옴표 제거)
    df = pd.read_csv(file_path, encoding='utf-8')
    df.columns = df.columns.str.strip()
    
    # 날짜 컬럼 정제
    df['날짜'] = df['날짜'].astype(str).str.replace(r'[\t"\s]', '', regex=True)
    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    df = df.dropna(subset=['날짜'])
    
    # 연도 추출 및 연평균 데이터 생성
    df['연도'] = df['날짜'].dt.year
    annual_df = df.groupby('연도').agg({
        '평균기온(℃)': 'mean',
        '최저기온(℃)': 'mean',
        '최고기온(℃)': 'mean'
    }).reset_index()
    
    return annual_df

# 데이터 불러오기
try:
    data_path = "ta_20260601093156.csv"
    df_annual = load_data(data_path)
except Exception as e:
    st.error(f"데이터 파일을 불러오는 중 오류가 발생했습니다: {e}")
    st.info("💡 팁: 'ta_20260601093156.csv' 파일이 스크립트와 같은 경로에 있는지 확인해주세요.")
    st.stop()

# 3. 사이드바 제어 요소
st.sidebar.header("📊 분석 설정")

# 지표 선택
target_col = st.sidebar.selectbox(
    "분석할 기온 지표 선택",
    ["평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]
)

# 기준 연도 선택 (기본값: 1980년)
split_year = st.sidebar.slider(
    "트렌드 분기 기준 연도",
    int(df_annual['연도'].min()), 
    int(df_annual['연도'].max()), 
    1980
)

# 이동평균선 선택
ma_window = st.sidebar.slider("이동평균선 주기 (년)", 5, 20, 10)
df_annual['이동평균'] = df_annual[target_col].rolling(window=ma_window, center=True).mean()

# 4. 데이터 분할 및 회귀분석 (기울기 계산)
df_before = df_annual[df_annual['연도'] < split_year]
df_after = df_annual[df_annual['연도'] >= split_year]

def get_trend_line(df, x_col, y_col):
    if len(df) < 2:
        return None, None, None
    slope, intercept = np.polyfit(df[x_col], df[y_col], 1)
    trend_y = slope * df[x_col] + intercept
    return slope, intercept, trend_y

slope_before, _, trend_before = get_trend_line(df_before, '연도', target_col)
slope_after, _, trend_after = get_trend_line(df_after, '연도', target_col)

# 5. 핵심 지표 메트릭 시각화
st.subheader("📈 구간별 기온 상승 속도 비교")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label=f"{df_annual['연도'].min()}년 ~ {split_year-1}년 (10년당 상승률)",
        value=f"{slope_before * 10:.3f} ℃" if slope_before else "데이터 부족"
    )
with col2:
    st.metric(
        label=f"{split_year}년 ~ {df_annual['연도'].max()}년 (10년당 상승률)",
        value=f"{slope_after * 10:.3f} ℃" if slope_after else "데이터 부족"
    )
with col3:
    if slope_before and slope_after:
        ratio = slope_after / slope_before if slope_before != 0 else 0
        st.metric(
            label="기준 연도 이후 상승 속도 변화",
            value=f"{ratio:.1f} 배 속도" if ratio > 0 else "역전됨",
            delta=f"{(slope_after - slope_before)*10:.3f} ℃ 더 가팔라짐"
        )

# 6. 메인 인터랙티브 차트 그리기 (Plotly)
fig = go.Figure()

# 실제 연평균 데이터 산점도
fig.add_trace(go.Scatter(
    x=df_annual['연도'], y=df_annual[target_col],
    mode='markers+lines', name='연평균 기온',
    line=dict(color='lightgray', width=1),
    marker=dict(color='darkgray', size=5)
))

# 이동평균선
fig.add_trace(go.Scatter(
    x=df_annual['연도'], y=df_annual['이동평균'],
    mode='lines', name=f'{ma_window}년 이동평균',
    line=dict(color='green', width=2, dash='dash')
))

# 이전 트렌드선
if slope_before is not None:
    fig.add_trace(go.Scatter(
        x=df_before['연도'], y=trend_before,
        mode='lines', name=f'{split_year}년 이전 트렌드',
        line=dict(color='blue', width=3)
    ))

# 이후 트렌드선
if slope_after is not None:
    fig.add_trace(go.Scatter(
        x=df_after['연도'], y=trend_after,
        mode='lines', name=f'{split_year}년 이후 트렌드',
        line=dict(color='red', width=3)
    ))

# 차트 레이아웃 설정
fig.update_layout(
    title=f"연도별 {target_col} 및 구간별 추세선 ({split_year}년 기준)",
    xaxis_title="연도",
    yaxis_title=target_col,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# 7. 데이터 상세 확인 탭
st.subheader("📋 데이터 세부 확인")
tab1, tab2 = st.tabs(["📊 데이터프레임 요약", "🧐 가설 검증 통찰"])

with tab1:
    st.dataframe(df_annual.style.format({
        '평균기온(℃)': '{:.2f}',
        '최저기온(℃)': '{:.2f}',
        '최고기온(℃)': '{:.2f}',
        '이동평균': '{:.2f}'
    }), use_container_width=True)

with tab2:
    st.markdown(f"""
    ### 💡 데이터가 말해주는 가설의 증거
    - **기울기 비교**: {split_year}년 이전에 비해 이후의 기온 상승 기울기(빨간선)가 더 가파른지 확인해보세요. 
    - **최저 기온 법칙**: 분석 지표를 **최저기온(℃)**으로 변경해 보세요. 보통 도시화 및 온난화 현상은 낮 기온(최고기온)보다 **밤 기온(최저기온)의 하한선을 끌어올리는 데** 더 큰 영향을 미칩니다.
    - **결론**: 만약 {split_year}년 이후의 '10년당 상승률'이 이전보다 유의미하게 높다면, 질문자님의 **"1980년대 이후 기온 상승세가 더 가팔라졌다"는 가설은 통계적으로 지지됩니다.**
    """)
