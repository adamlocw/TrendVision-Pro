import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 1. Configuration & Translations / 設定與翻譯
# -----------------------------------------------------------------------------

# 更新專案名稱為 TrendVision Pro
st.set_page_config(layout="wide", page_title="TrendVision Pro")

TRANSLATIONS = {
    'en': {
        'title': 'TrendVision Pro: Trendlines & MACD',
        'sidebar_title': 'Configuration',
        'select_ticker_label': 'Select Ticker',
        'custom_ticker_label': 'Or Enter Custom Ticker',
        'period_label': 'Select Time Period',
        'year_unit': 'Year(s)',
        'month_1': '1 Month',
        'month_3': '3 Months',
        'month_6': '6 Months',
        'language_label': 'Select Language',
        'error_fetch': 'Error fetching data. Please check the ticker symbol.',
        'data_success': 'Data loaded successfully for',
        'tab_charts': 'Charts & Analysis',
        'tab_data': 'Raw Data',
        'macd_bullish': 'BULLISH DIVERGENCE DETECTED',
        'macd_bearish': 'BEARISH DIVERGENCE DETECTED',
        'trend_breakout': 'BULLISH BREAKOUT (Ascending)',
        'trend_breakdown': 'BEARISH BREAKDOWN (Descending)',
        'price': 'Price',
        'volume': 'Volume',
        'resistance': 'Resistance Trendline',
        'support': 'Support Trendline',
        'detected_patterns': 'Detected Patterns (Last 90 Days)',
        'no_patterns': 'No significant patterns detected recently.',
        'instructions': 'Select a ticker or enter a custom one (e.g., AMD, INTC).'
    },
    'tc': {
        'title': 'TrendVision Pro：趨勢線與 MACD 背馳',
        'sidebar_title': '設定',
        'select_ticker_label': '選擇代號',
        'custom_ticker_label': '或輸入自訂代號',
        'period_label': '選擇時間範圍',
        'year_unit': '年',
        'month_1': '1 個月',
        'month_3': '3 個月',
        'month_6': '6 個月',
        'language_label': '選擇語言',
        'error_fetch': '獲取數據時發生錯誤，請檢查代號是否正確。',
        'data_success': '成功載入數據：',
        'tab_charts': '圖表分析',
        'tab_data': '原始數據',
        'macd_bullish': '偵測到 MACD 看漲背馳 (底背離)',
        'macd_bearish': '偵測到 MACD 看跌背馳 (頂背離)',
        'trend_breakout': '向上突破下降趨勢線 (看漲)',
        'trend_breakdown': '跌破上升趨勢線 (看跌)',
        'price': '價格',
        'volume': '成交量',
        'resistance': '壓力線 (下降趨勢)',
        'support': '支撐線 (上升趨勢)',
        'detected_patterns': '近期偵測到的型態 (過去 90 天)',
        'no_patterns': '近期未偵測到顯著型態。',
        'instructions': '選擇代號或自行輸入 (例如 AMD, INTC)。'
    }
}

# Default Ticker List / 預設清單 (Crypto + Mag 7)
DEFAULT_TICKERS = [
    "BTC-USD", "ETH-USD", "SOL-USD", 
    "NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "META"
]

# -----------------------------------------------------------------------------
# 2. Helper Functions / 輔助功能
# -----------------------------------------------------------------------------

def get_text(key, lang_code):
    return TRANSLATIONS[lang_code].get(key, key)

@st.cache_data(ttl=300)
def fetch_data(ticker, days):
    """Fetch data with variable lookback period"""
    try:
        # Fetch slightly more than requested to ensure indicators stabilize
        # We add a buffer (e.g., 50 days) for EMA calculations
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 50)
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        
        # Flatten MultiIndex columns if present (common issue with new yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna()
        return df
    except Exception as e:
        return None

def calculate_macd(df):
    """Calculate MACD, Signal, and Histogram"""
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    return df

# -----------------------------------------------------------------------------
# 3. Technical Analysis Logic / 技術指標邏輯
# -----------------------------------------------------------------------------

def find_divergence(df, lookback=5, order=3):
    """
    Detects simple regular divergence.
    """
    df = df.copy()
    
    # Find local peaks (highs) and troughs (lows) for Price and MACD
    price_highs = argrelextrema(df['Close'].values, np.greater, order=order)[0]
    price_lows = argrelextrema(df['Close'].values, np.less, order=order)[0]
    
    macd_highs = argrelextrema(df['MACD'].values, np.greater, order=order)[0]
    macd_lows = argrelextrema(df['MACD'].values, np.less, order=order)[0]
    
    bullish_divs = []
    bearish_divs = []
    
    # Logic for Bearish Divergence (Price makes Higher High, MACD makes Lower High)
    if len(price_highs) >= 2 and len(macd_highs) >= 2:
        p2, p1 = price_highs[-1], price_highs[-2]
        
        if p2 > len(df) - 60:
            price_higher_high = df['Close'].iloc[p2] > df['Close'].iloc[p1]
            
            m2_idx = min(macd_highs, key=lambda x: abs(x-p2))
            m1_idx = min(macd_highs, key=lambda x: abs(x-p1))
            
            if m1_idx != m2_idx:
                macd_lower_high = df['MACD'].iloc[m2_idx] < df['MACD'].iloc[m1_idx]
                
                if price_higher_high and macd_lower_high:
                    bearish_divs.append(df.index[p2])

    # Logic for Bullish Divergence (Price makes Lower Low, MACD makes Higher Low)
    if len(price_lows) >= 2 and len(macd_lows) >= 2:
        p2, p1 = price_lows[-1], price_lows[-2]
        
        if p2 > len(df) - 60:
            price_lower_low = df['Close'].iloc[p2] < df['Close'].iloc[p1]
            
            m2_idx = min(macd_lows, key=lambda x: abs(x-p2))
            m1_idx = min(macd_lows, key=lambda x: abs(x-p1))
            
            if m1_idx != m2_idx:
                macd_higher_low = df['MACD'].iloc[m2_idx] > df['MACD'].iloc[m1_idx]
                
                if price_lower_low and macd_higher_low:
                    bullish_divs.append(df.index[p2])
                    
    return bullish_divs, bearish_divs

def find_trendlines(df, order=5):
    """
    Identifies Resistance (connecting highs) and Support (connecting lows) lines.
    """
    df['id'] = range(len(df))
    high_idx = argrelextrema(df['High'].values, np.greater, order=order)[0]
    low_idx = argrelextrema(df['Low'].values, np.less, order=order)[0]
    
    trendlines = []
    
    # --- Resistance Line (Connecting Highs) ---
    if len(high_idx) >= 2:
        x_highs = df['id'].iloc[high_idx[-2:]].values
        y_highs = df['High'].iloc[high_idx[-2:]].values
        
        if len(x_highs) == 2:
            coeffs_res = np.polyfit(x_highs, y_highs, 1)
            slope_res, intercept_res = coeffs_res
            
            curr_x = df['id'].iloc[-1]
            curr_close = df['Close'].iloc[-1]
            line_val = slope_res * curr_x + intercept_res
            
            trendlines.append({
                'type': 'resistance',
                'slope': slope_res,
                'intercept': intercept_res,
                'x_vals': [df.index[high_idx[-2]], df.index[-1]],
                'y_vals': [y_highs[0], line_val],
                'breakout': curr_close > line_val and slope_res < 0
            })

    # --- Support Line (Connecting Lows) ---
    if len(low_idx) >= 2:
        x_lows = df['id'].iloc[low_idx[-2:]].values
        y_lows = df['Low'].iloc[low_idx[-2:]].values
        
        if len(x_lows) == 2:
            coeffs_sup = np.polyfit(x_lows, y_lows, 1)
            slope_sup, intercept_sup = coeffs_sup
            
            curr_x = df['id'].iloc[-1]
            curr_close = df['Close'].iloc[-1]
            line_val = slope_sup * curr_x + intercept_sup
            
            trendlines.append({
                'type': 'support',
                'slope': slope_sup,
                'intercept': intercept_sup,
                'x_vals': [df.index[low_idx[-2]], df.index[-1]],
                'y_vals': [y_lows[0], line_val],
                'breakdown': curr_close < line_val and slope_sup > 0
            })
            
    return trendlines

# -----------------------------------------------------------------------------
# 4. Main Application Logic / 主程式
# -----------------------------------------------------------------------------

def main():
    # Sidebar
    st.sidebar.title("📊 TrendVision Pro")
    
    lang = st.sidebar.radio(
        "Language / 語言",
        ('English', '繁體中文'),
        index=0
    )
    lang_code = 'tc' if lang == '繁體中文' else 'en'
    
    st.sidebar.header(get_text('sidebar_title', lang_code))
    
    # Dropdown selection
    selected_option = st.sidebar.selectbox(
        get_text('select_ticker_label', lang_code),
        options=DEFAULT_TICKERS + ["Other / 其他"]
    )

    if selected_option == "Other / 其他":
        ticker_input = st.sidebar.text_input(
            get_text('custom_ticker_label', lang_code), 
            value="BTC-USD"
        ).upper()
    else:
        ticker_input = selected_option
    
    st.sidebar.caption(get_text('instructions', lang_code))

    # Time Period Selection
    # Values represent years. 1/12 is approx 1 month.
    time_options = [1/12, 0.25, 0.5, 1, 2, 3, 4, 5]
    
    def format_time_option(val):
        if val == 1/12:
            return get_text('month_1', lang_code)
        if val == 0.25:
            return get_text('month_3', lang_code)
        if val == 0.5:
            return get_text('month_6', lang_code)
        return f"{val} {get_text('year_unit', lang_code)}"

    selected_val = st.sidebar.selectbox(
        get_text('period_label', lang_code),
        options=time_options,
        format_func=format_time_option,
        index=3  # Index 3 corresponds to '1' in the list (Default 1 year)
    )
    days_lookback = int(selected_val * 365)
    
    st.title(f"{ticker_input} - {get_text('title', lang_code)}")

    # Fetch Data
    with st.spinner('Loading data...'):
        df = fetch_data(ticker_input, days_lookback)

    if df is None or len(df) < 30:
        st.error(get_text('error_fetch', lang_code))
        return

    st.success(f"{get_text('data_success', lang_code)} {ticker_input}")

    # Process Indicators
    df = calculate_macd(df)
    bullish_divs, bearish_divs = find_divergence(df)
    trendlines = find_trendlines(df)

    # Tabs
    tab1, tab2 = st.tabs([get_text('tab_charts', lang_code), get_text('tab_data', lang_code)])

    with tab1:
        # Create Plotly Figure
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05, 
            row_heights=[0.7, 0.3],
            subplot_titles=(get_text('price', lang_code), 'MACD')
        )

        # 1. Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='OHLC'
        ), row=1, col=1)

        # 2. Add Trendlines
        alerts = []
        for line in trendlines:
            color = 'red' if line['type'] == 'resistance' else 'green'
            name = get_text('resistance', lang_code) if line['type'] == 'resistance' else get_text('support', lang_code)
            
            fig.add_trace(go.Scatter(
                x=line['x_vals'], 
                y=line['y_vals'],
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                name=name
            ), row=1, col=1)

            # Check Breakouts for Alerts
            if line.get('breakout'):
                alerts.append(f"🟢 {get_text('trend_breakout', lang_code)}")
            if line.get('breakdown'):
                alerts.append(f"🔴 {get_text('trend_breakdown', lang_code)}")

        # 3. Add MACD
        fig.add_trace(go.Bar(x=df.index, y=df['Hist'], name='Histogram', marker_color='gray'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='orange')), row=2, col=1)

        # 4. Highlight Divergences
        for date in bullish_divs:
            alerts.append(f"🚀 {get_text('macd_bullish', lang_code)}: {date.strftime('%Y-%m-%d')}")
            val = df.loc[date, 'Low']
            fig.add_annotation(x=date, y=val, text="Bull Div", showarrow=True, arrowhead=1, ax=0, ay=30, row=1, col=1)
            
        for date in bearish_divs:
            alerts.append(f"📉 {get_text('macd_bearish', lang_code)}: {date.strftime('%Y-%m-%d')}")
            val = df.loc[date, 'High']
            fig.add_annotation(x=date, y=val, text="Bear Div", showarrow=True, arrowhead=1, ax=0, ay=-30, row=1, col=1)

        # Update Layout
        fig.update_layout(
            height=800, 
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Display Alerts
        st.subheader(get_text('detected_patterns', lang_code))
        if alerts:
            for alert in alerts:
                st.write(alert)
        else:
            st.info(get_text('no_patterns', lang_code))

    with tab2:
        st.dataframe(df.sort_index(ascending=False).head(100))

if __name__ == "__main__":
    main()
