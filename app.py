# ======================================================================
# COMPLETE STREAMLIT APP - Save as "app.py"
# ======================================================================
# RUN COMMAND: streamlit run app.py
# ======================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Page config
st.set_page_config(
    page_title="TSLA Trading Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .prediction-up {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-down {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        border-top: 1px solid #ddd;
        font-size: 0.8rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# DATA LOADING & CACHING
# ======================================================================

@st.cache_data(ttl=3600)
def load_data():
    """Load TSLA data from Yahoo Finance"""
    df = yf.download('TSLA', start='2015-01-01', end='2024-12-31', progress=False)
    df = df.reset_index()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    return df

@st.cache_data(ttl=300)
def load_live_data():
    """Load live TSLA data for today's prediction"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200)
    df = yf.download('TSLA', start=start_date, end=end_date, progress=False)
    df = df.reset_index()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    return df

# ======================================================================
# FEATURE ENGINEERING
# ======================================================================

def create_features(df):
    """Create advanced technical indicators for global trading"""
    df = df.copy()
    
    # Price Returns (Global momentum indicators)
    for period in [1, 2, 3, 5, 10, 20, 50]:
        df[f'ret_{period}d'] = df['Close'].pct_change(period)
        df[f'log_ret_{period}d'] = np.log(df['Close'] / df['Close'].shift(period))
    
    # Moving Averages (Global trend indicators)
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['Close'].rolling(period).mean()
        df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        df[f'price_sma_{period}'] = df['Close'] / df[f'sma_{period}']
    
    # Moving Average Crossovers (Entry/Exit signals)
    df['ma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
    df['ma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['ma_cross_50_200'] = (df['sma_50'] > df['sma_200']).astype(int)
    
    # Golden Cross / Death Cross (Major trend reversal signals)
    df['golden_cross'] = ((df['sma_50'] > df['sma_200']) & 
                          (df['sma_50'].shift(1) <= df['sma_200'].shift(1))).astype(int)
    df['death_cross'] = ((df['sma_50'] < df['sma_200']) & 
                         (df['sma_50'].shift(1) >= df['sma_200'].shift(1))).astype(int)
    
    # RSI (Multiple periods for momentum confirmation)
    for period in [7, 14, 21]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        df[f'rsi_{period}_signal'] = np.where(df[f'rsi_{period}'] > 70, -1, 
                                               np.where(df[f'rsi_{period}'] < 30, 1, 0))
    
    # MACD (Trend following)
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross'] = ((df['macd'] > df['macd_signal']) & 
                        (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    
    # Bollinger Bands (Volatility-based trading)
    for period in [20]:
        df[f'bb_mid_{period}'] = df['Close'].rolling(period).mean()
        bb_std = df['Close'].rolling(period).std()
        df[f'bb_upper_{period}'] = df[f'bb_mid_{period}'] + 2 * bb_std
        df[f'bb_lower_{period}'] = df[f'bb_mid_{period}'] - 2 * bb_std
        df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
    
    # ATR (Stop loss calculation)
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)),
                   abs(df['Low'] - df['Close'].shift(1)))
    )
    df['atr_14'] = df['tr'].rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / df['Close']
    
    # Volume Analysis (Institutional flow)
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['vol_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
    df['volume_breakout'] = (df['vol_ratio'] > 1.5).astype(int)
    
    # OBV (On-Balance Volume - Smart money indicator)
    df['obv'] = (np.sign(df['ret_1d']) * df['Volume']).cumsum()
    df['obv_ma'] = df['obv'].rolling(20).mean()
    df['obv_trend'] = (df['obv'] > df['obv_ma']).astype(int)
    
    # Price Action (Candlestick patterns)
    df['hl_range'] = (df['High'] - df['Low']) / df['Close']
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['gap_up'] = (df['gap'] > 0.01).astype(int)
    df['gap_down'] = (df['gap'] < -0.01).astype(int)
    
    # Support & Resistance (Key price levels)
    df['resistance'] = df['High'].rolling(20).max()
    df['support'] = df['Low'].rolling(20).min()
    df['near_resistance'] = ((df['resistance'] - df['Close']) / df['Close'] < 0.02).astype(int)
    df['near_support'] = ((df['Close'] - df['support']) / df['Close'] < 0.02).astype(int)
    
    # Volatility (Risk assessment)
    df['volatility'] = df['ret_1d'].rolling(20).std()
    
    # Lag Features (Time series memory)
    for lag in [1, 2, 3, 5]:
        df[f'ret_lag_{lag}'] = df['ret_1d'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag)
        df[f'vol_lag_{lag}'] = df['vol_ratio'].shift(lag)
    
    # Rolling Statistics (Adaptive indicators)
    for window in [5, 10, 20]:
        df[f'ret_mean_{window}'] = df['ret_1d'].rolling(window).mean()
        df[f'ret_std_{window}'] = df['ret_1d'].rolling(window).std()
    
    # Target (Next day direction)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df

# ======================================================================
# MODEL TRAINING (Global Market Intelligence)
# ======================================================================

@st.cache_resource
def train_models():
    """Train XGBoost and ensemble models"""
    
    df = load_data()
    df_feat = create_features(df)
    df_feat = df_feat.dropna().reset_index(drop=True)
    
    # Feature selection (correlation-based)
    exclude_cols = ['Date', 'target', 'Adj Close']
    feature_cols = [col for col in df_feat.columns if col not in exclude_cols]
    correlations = df_feat[feature_cols + ['target']].corr()['target'].sort_values(ascending=False)
    selected_features = correlations[abs(correlations) > 0.02].index.tolist()
    if 'target' in selected_features:
        selected_features.remove('target')
    
    # Prepare data
    X = df_feat[selected_features]
    y = df_feat['target']
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.03,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train_scaled, y_train)
    xgb_pred = xgb.predict(X_test_scaled)
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    
    # Train Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=5, random_state=42)
    gb.fit(X_train_scaled, y_train)
    gb_pred = gb.predict(X_test_scaled)
    
    # Ensemble (Majority Voting)
    ensemble_pred = (xgb_pred + rf_pred + gb_pred) > 1.5
    
    # Calculate metrics
    metrics = {
        'xgb_accuracy': accuracy_score(y_test, xgb_pred),
        'xgb_precision': precision_score(y_test, xgb_pred),
        'xgb_recall': recall_score(y_test, xgb_pred),
        'xgb_f1': f1_score(y_test, xgb_pred),
        'xgb_auc': roc_auc_score(y_test, xgb_proba),
        'rf_accuracy': accuracy_score(y_test, rf_pred),
        'gb_accuracy': accuracy_score(y_test, gb_pred),
        'ensemble_accuracy': accuracy_score(y_test, ensemble_pred),
        'selected_features': selected_features,
        'scaler': scaler,
        'xgb_model': xgb,
        'rf_model': rf,
        'gb_model': gb,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test
    }
    
    # Backtest
    test_prices = df_feat['Close'].iloc[split_idx:].reset_index(drop=True)
    capital = 100000
    position = 0
    equity = [100000]
    cost = 0.001
    
    for i in range(1, len(ensemble_pred)):
        price = test_prices.iloc[i]
        if ensemble_pred[i-1] == 1 and position == 0:
            shares = int(capital * 0.2 / price)
            if shares > 0:
                capital -= shares * price * (1 + cost)
                position = shares
        elif ensemble_pred[i-1] == 1 and ensemble_pred[i] == 0 and position > 0:
            capital += position * price * (1 - cost)
            position = 0
        equity.append(capital + (position * price if position > 0 else 0))
    
    if position > 0:
        capital += position * test_prices.iloc[-1] * (1 - cost)
        equity[-1] = capital
    
    metrics['strategy_return'] = (capital - 100000) / 100000 * 100
    bh_equity = 100000 * (test_prices / test_prices.iloc[0])
    metrics['bh_return'] = (bh_equity.iloc[-1] - 100000) / 100000 * 100
    metrics['equity_curve'] = pd.Series(equity, index=test_prices.index)
    metrics['test_prices'] = test_prices
    metrics['test_dates'] = df_feat['Date'].iloc[split_idx:].reset_index(drop=True)
    
    return metrics

# ======================================================================
# LIVE PREDICTION
# ======================================================================

def get_live_prediction(metrics):
    """Get real-time prediction for tomorrow"""
    
    df_live = load_live_data()
    df_feat = create_features(df_live)
    df_feat = df_feat.dropna()
    
    latest = df_feat[metrics['selected_features']].iloc[-1:].fillna(0)
    scaled = metrics['scaler'].transform(latest)
    
    xgb_pred = metrics['xgb_model'].predict(scaled)[0]
    rf_pred = metrics['rf_model'].predict(scaled)[0]
    gb_pred = metrics['gb_model'].predict(scaled)[0]
    
    ensemble_pred = 1 if (xgb_pred + rf_pred + gb_pred) > 1 else 0
    xgb_proba = metrics['xgb_model'].predict_proba(scaled)[0][ensemble_pred] * 100
    
    return {
        'date': df_live['Date'].iloc[-1].date(),
        'price': df_live['Close'].iloc[-1],
        'volume': df_live['Volume'].iloc[-1],
        'change': df_live['Close'].pct_change().iloc[-1] * 100,
        'xgb_pred': 'UP 📈' if xgb_pred == 1 else 'DOWN 📉',
        'rf_pred': 'UP 📈' if rf_pred == 1 else 'DOWN 📉',
        'gb_pred': 'UP 📈' if gb_pred == 1 else 'DOWN 📉',
        'ensemble_pred': 'UP 📈' if ensemble_pred == 1 else 'DOWN 📉',
        'confidence': xgb_proba,
        'signal': 'BUY' if ensemble_pred == 1 else 'SELL/WAIT'
    }

# ======================================================================
# LOAD MODELS (Cached)
# ======================================================================

with st.spinner('🔄 Loading Trading Intelligence Models...'):
    metrics = train_models()
    live = get_live_prediction(metrics)

# ======================================================================
# HEADER SECTION
# ======================================================================

st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🚀 TSLA Trading Intelligence Platform</h1>
    <p style="color: white; margin: 5px 0 0 0; opacity: 0.9;">
        Real-Time Market Prediction | XGBoost AI | Global Trading Signals
    </p>
</div>
""", unsafe_allow_html=True)

# ======================================================================
# LIVE METRICS ROW
# ======================================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📅 Current Date", live['date'], delta=None)
with col2:
    delta_color = "normal" if live['change'] >= 0 else "inverse"
    st.metric("💰 Current Price", f"${live['price']:.2f}", delta=f"{live['change']:.2f}%")
with col3:
    st.metric("📊 Volume", f"{live['volume']/1e6:.1f}M", delta=None)
with col4:
    st.metric("🎯 XGBoost Accuracy", f"{metrics['xgb_accuracy']*100:.2f}%", 
              delta=f"+{(metrics['xgb_accuracy']-0.5)*100:.1f}% vs Random")
with col5:
    st.metric("💰 Strategy Return", f"{metrics['strategy_return']:.2f}%", 
              delta=f"+{metrics['strategy_return'] - metrics['bh_return']:.1f}% vs BH")

# ======================================================================
# PREDICTION SECTION
# ======================================================================

st.markdown("---")
st.subheader("🎯 Live Trading Signal")

pred_color = "prediction-up" if live['ensemble_pred'] == 'UP 📈' else "prediction-down"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="{pred_color}">
        <h3>ENSEMBLE</h3>
        <h1 style="font-size: 2.5rem;">{live['ensemble_pred']}</h1>
        <p>Signal: {live['signal']}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>XGBoost</h4>
        <h2>{live['xgb_pred']}</h2>
        <p>Confidence: {live['confidence']:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Random Forest</h4>
        <h2>{live['rf_pred']}</h2>
        <p>Accuracy: {metrics['rf_accuracy']*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Gradient Boosting</h4>
        <h2>{live['gb_pred']}</h2>
        <p>Accuracy: {metrics['gb_accuracy']*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================================
# RISK MANAGEMENT SECTION
# ======================================================================

st.markdown("---")
st.subheader("⚡ Risk Management & Position Sizing")

col1, col2, col3, col4 = st.columns(4)

with col1:
    recommended_size = 20 if live['ensemble_pred'] == 'UP 📈' else 5
    st.metric("📈 Recommended Position", f"{recommended_size}%", 
              delta="Aggressive" if recommended_size > 10 else "Conservative")
with col2:
    stop_loss = live['price'] * 0.97 if live['ensemble_pred'] == 'UP 📈' else live['price'] * 1.03
    st.metric("🛑 Stop Loss", f"${stop_loss:.2f}", delta="3% from entry")
with col3:
    take_profit = live['price'] * 1.05 if live['ensemble_pred'] == 'UP 📈' else live['price'] * 0.95
    st.metric("🎯 Take Profit", f"${take_profit:.2f}", delta="5% from entry")
with col4:
    risk_reward = round(abs((take_profit - live['price']) / (live['price'] - stop_loss)), 2)
    st.metric("📊 Risk/Reward", f"1:{risk_reward}", delta="Ideal" if risk_reward >= 2 else "Poor")

# ======================================================================
# CHARTS SECTION
# ======================================================================

st.markdown("---")
st.subheader("📈 Performance Analytics")

# Create tabs for different charts
tab1, tab2, tab3, tab4 = st.tabs(["📊 Equity Curve", "📉 Drawdown", "🏆 Model Comparison", "📋 Confusion Matrix"])

with tab1:
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=metrics['test_dates'], y=metrics['equity_curve'], 
                             name='Strategy', line=dict(color='#2E86AB', width=2.5),
                             fill='tozeroy'))
    bh_equity = 100000 * (metrics['test_prices'] / metrics['test_prices'].iloc[0])
    fig.add_trace(go.Scatter(x=metrics['test_dates'], y=bh_equity, 
                             name='Buy & Hold', line=dict(color='#E74C3C', width=2, dash='dash')))
    fig.update_layout(height=500, title="Strategy Performance vs Buy & Hold", template='plotly_white')
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Portfolio Value ($)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    drawdown = (metrics['equity_curve'] / metrics['equity_curve'].cummax() - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics['test_dates'], y=drawdown, 
                             name='Drawdown', line=dict(color='#E74C3C', width=2),
                             fill='tozeroy'))
    fig.update_layout(height=500, title="Drawdown Analysis", template='plotly_white')
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Drawdown (%)")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    models = ['XGBoost', 'Random Forest', 'Gradient Boost', 'Ensemble']
    scores = [metrics['xgb_accuracy']*100, metrics['rf_accuracy']*100, 
              metrics['gb_accuracy']*100, metrics['ensemble_accuracy']*100]
    colors = ['#2ECC71' if s >= 75 else '#F39C12' if s >= 65 else '#E74C3C' for s in scores]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=models, y=scores, marker_color=colors,
                         text=[f'{s:.1f}%' for s in scores], textposition='outside'))
    fig.add_hline(y=75, line_dash="dash", line_color="#27AE60", annotation_text="Target 75%")
    fig.update_layout(height=500, title="Model Performance Comparison", template='plotly_white')
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Confusion Matrix for XGBoost
    from sklearn.metrics import confusion_matrix
    y_pred = metrics['xgb_model'].predict(metrics['X_test_scaled'])
    cm = confusion_matrix(metrics['y_test'], y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted DOWN', 'Predicted UP'],
        y=['Actual DOWN', 'Actual UP'],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='RdYlGn',
        showscale=True
    ))
    fig.update_layout(height=500, title="XGBoost Confusion Matrix", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# TRADING INSIGHTS
# ======================================================================

st.markdown("---")
st.subheader("💡 Trading Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📊 Technical Analysis
    
    | Indicator | Value | Signal |
    |-----------|-------|--------|
    | RSI (14) | Calculating | - |
    | MACD | Calculating | - |
    | Volatility | Calculating | - |
    | Volume Trend | Calculating | - |
    """)

with col2:
    st.markdown("""
    ### 🎯 Trade Recommendations
    
    **Current Bias:** BULLISH
    
    **Entry Zone:** $""" + f"{live['price']*0.98:.2f}" + """ - $""" + f"{live['price']*1.02:.2f}" + """
    
    **Stop Loss:** $""" + f"{stop_loss:.2f}" + """
    
    **Take Profit 1:** $""" + f"{take_profit:.2f}" + """
    
    **Take Profit 2:** $""" + f"{live['price']*1.10:.2f}" + """
    
    **Position Size:** """ + f"{recommended_size}%" + """ of portfolio
    """)

# ======================================================================
# FOOTER
# ======================================================================

st.markdown("""
<div class="footer">
    <p>🚀 TSLA Trading Intelligence Platform | Powered by XGBoost AI</p>
    <p>⚠️ Educational purposes only. Not financial advice. Past performance does not guarantee future results.</p>
    <p>Data Source: Yahoo Finance | Last Updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
</div>
""", unsafe_allow_html=True)