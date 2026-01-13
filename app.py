import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# -----------------------------------------------------------------------------
# 0. CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NVDA Option Analysis Case Study",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed" 
)

# Custom CSS for Clean, Colorful, Professional UI
# Palette: Deep Navy (#1E3A8A), Emerald (#10B981), Crimson (#EF4444), Amber (#F59E0B), Royal Blue (#2563EB)
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #F8FAFC; /* slightly cool gray/white */
        color: #1F2937;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headings */
    h1 {
        color: #1E3A8A !important; /* Deep Navy */
        font-weight: 800;
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 5px solid #2563EB; /* Royal Blue */
        margin-bottom: 30px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    h2 {
        color: #1E40AF !important;
        font-weight: 700;
        border-left: 6px solid #F59E0B; /* Amber accent */
        padding-left: 15px;
        margin-top: 40px;
        background: linear-gradient(90deg, #EFF6FF 0%, transparent 100%);
        padding-top: 10px;
        padding-bottom: 10px;
        border-radius: 0 10px 10px 0;
    }
    h3 {
        color: #111827 !important;
        font-weight: 600;
        margin-top: 25px;
    }
    h4 {
        color: #1F2937 !important;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    
    /* Text */
    p, li, .stMarkdown {
        font-size: 1.1rem !important;
        line-height: 1.8 !important;
        color: #374151;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: #2563EB;
    }
    div[data-testid="stMetric"] label {
        color: #6B7280;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1E3A8A;
        font-size: 2rem !important;
        font-weight: 800;
    }
    
    /* Tabs - Horizontal Scrolling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        padding-bottom: 10px;
        border-bottom: 3px solid #E5E7EB;
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
        display: flex;
        flex-wrap: nowrap;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        color: #4B5563;
        border-radius: 8px;
        padding: 12px 24px;
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        transition: all 0.2s;
        flex-shrink: 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #ffffff !important;
        border-color: #2563EB !important;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
    }
    
    /* Tab Content - Normal Vertical Scrolling */
    .stTabs [data-baseweb="tab-panel"] {
        overflow-y: visible;
        overflow-x: visible;
    }
    
    /* Custom Info Box */
    .edu-box {
        background-color: #F0F9FF; /* Light Blue */
        border-left: 5px solid #0EA5E9; /* Sky Blue */
        padding: 25px;
        margin: 20px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .theory-header {
        color: #0369A1;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 10px;
        display: block;
    }
    
    /* Concept Box */
    .concept-box {
        background-color: #ECFDF5; /* Light Emerald */
        border-left: 5px solid #10B981; /* Emerald */
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .concept-title {
        color: #047857;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.9rem;
        margin-bottom: 5px;
        display: block;
    }

    /* Warning Box */
    .risk-box {
        background-color: #FEF2F2; /* Light Red */
        border-left: 5px solid #EF4444; /* Red */
        padding: 20px;
        border-radius: 8px;
    }

    /* Remove Sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* LaTeX Styling */
    .katex {
        font-size: 1.2em;
        color: #111827;
    }
    
    /* Memo Box */
    .memo-box {
        background-color: #fff;
        border: 2px solid #ddd;
        padding: 30px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
        font-family: "Courier New", monospace;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. DATA & MODEL UTILITIES
# -----------------------------------------------------------------------------
@st.cache_data
def get_nvda_data():
    """Fetches last 12 months of NVDA daily data."""
    end = datetime.today()
    start = end - timedelta(days=365)
    data = yf.download("NVDA", start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Moving Averages for Chart
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    return data

class BSModel:
    """Black-Scholes Pricing & Greeks"""
    def __init__(self, S, K, T, r, sigma, opt_type="call"):
        self.S, self.K, self.T, self.r, self.sigma = S, K, T, r, sigma
        self.type = opt_type.lower()
        self.d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        self.d2 = self.d1 - sigma*np.sqrt(T)
        
    def price(self):
        if self.type == "call":
            return self.S*si.norm.cdf(self.d1) - self.K*np.exp(-self.r*self.T)*si.norm.cdf(self.d2)
        else:
            return self.K*np.exp(-self.r*self.T)*si.norm.cdf(-self.d2) - self.S*si.norm.cdf(-self.d1)
            
    def delta(self):
        return si.norm.cdf(self.d1) if self.type == "call" else -si.norm.cdf(-self.d1)
        
    def theta(self):
        term1 = -(self.S * si.norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        term2 = -self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2 if self.type=="call" else -self.d2)
        return (term1 + term2) / 365.0 # Daily
        
    def vega(self):
        return (self.S * np.sqrt(self.T) * si.norm.pdf(self.d1)) / 100.0 # 1% change

# Load Data Once
df = get_nvda_data()
current_price = df['Close'].iloc[-1]
hist_vol = df['Log_Ret'].std() * np.sqrt(252)

# FIXED ASSUMPTIONS (Strict Case Study)
ASSUMED_S = current_price # Real market data
ASSUMED_K = round(current_price * 1.10, 0) # 10% OTM Strike (rounded)
ASSUMED_T = 0.25 # 3 Months
ASSUMED_R = 0.05 # 5% Risk Free
ASSUMED_SIGMA = hist_vol

# -----------------------------------------------------------------------------
# 2. MAIN HEADER
# -----------------------------------------------------------------------------
st.title("Financial Derivatives Case Study: NVIDIA (NVDA)")
st.markdown("""
<div style='text-align: center; color: #4B5563; padding-bottom: 20px; font-size: 1.2rem;'>
    <strong>Senior Quantitative Analysis Report</strong> | Black-Scholes Valuation & Risk Assessment
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. CONTENT TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "Overview", "Market Data", "Volatility", "Pricing Model", "Greeks", 
    "Risk & Hedging", "Sensitivity", "Call vs Put", "Limitations", "Interactive Simulator"
])

# COLORS
C_BLUE = '#2563EB'
C_GREEN = '#10B981'
C_RED = '#EF4444'
C_AMBER = '#F59E0B'
C_DARK = '#1F2937'

# --- TAB 1: OVERVIEW ---
with tabs[0]:
    st.header("1. Executive Overview & Objective")
    
    st.markdown("""
    <div class='edu-box'>
        <span class='theory-header'>📘 Theory: Derivatives 101</span>
        <p>A <strong>Financial Derivative</strong> is a contract whose value is "derived" from the performance of an underlying asset (in this case, NVIDIA stock). We are specifically analyzing a <strong>European Call Option</strong>.</p>
        <ul>
            <li><strong>Call Option:</strong> Gives the holder the right (but not the obligation) to BUY the stock at a specific price (Strike) by a specific date (Expiry).</li>
            <li><strong>European:</strong> Can only be exercised AT the expiration date (unlike American options which can be exercised anytime).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 The Objective")
        st.write("""
        We are acting as a **Quantitative Analyst**. Our goal is to:
        1.  Determine the **"Fair Value"** of a 3-month Call Option on NVDA.
        2.  Assess the **Risks** involved in holding this option.
        3.  Propose a **Hedging Strategy** to eliminate those risks if needed.
        """)
    with col2:
        st.markdown("### 🔑 Key Terminology")
        st.markdown(f"""
        -   **Underlying ($S$)**: NVIDIA Stock Price.
        -   **Strike Price ($K$)**: The target price we are betting NVDA will exceed.
        -   **Premium**: The price we pay today to buy this option contract.
        -   **Bullish**: We profit if NVDA goes **UP**.
        """)

    st.markdown("### 💡 Why do we care?")
    st.write("""
    Options provide **leverage**. A 10% move in the stock price might result in a 50% or 100% gain in the option price. However, this comes with the risk of losing 100% of the premium if the stock doesn't move as expected.
    """)

# --- TAB 2: DATA ---
with tabs[1]:
    st.header("2. Data Collection: NVIDIA (NVDA)")
    
    st.markdown("""
    <div class='edu-box'>
        <span class='theory-header'>📊 Theory: Technical Analysis & Trend</span>
        <p>Before pricing the derivative, we must understand the asset. We use <strong>Candlestick Charts</strong> and <strong>Moving Averages</strong> to assess the trend.</p>
        <ul>
            <li><strong>SMA-50 (Short Term Trend):</strong> The average price over the last 50 days. If the price is above this, the short-term trend is UP.</li>
            <li><strong>SMA-200 (Long Term Trend):</strong> The average price over the last 200 days. This is a major support/resistance level for institutions.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced Charting (Candlestick + Volume)
    fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, subplot_titles=(f'NVDA Price Action', 'Trading Volume'), 
                               row_width=[0.2, 0.7])
    
    # Price
    fig_candle.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC',
                                        increasing_line_color=C_GREEN, decreasing_line_color=C_RED), row=1, col=1)
    fig_candle.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color=C_BLUE, width=2), name='50-Day SMA'), row=1, col=1)
    fig_candle.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color=C_AMBER, width=2), name='200-Day SMA'), row=1, col=1)
    
    # Volume
    colors = [C_GREEN if row['Open'] - row['Close'] >= 0 else C_RED for index, row in df.iterrows()]
    fig_candle.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
    
    fig_candle.update_layout(xaxis_rangeslider_visible=False, template="plotly_white", height=700, 
                             margin=dict(l=50, r=50, t=50, b=50), hovermode="x unified")
    st.plotly_chart(fig_candle, use_container_width=True)
    
    st.markdown("### 📝 Observation")
    st.write(f"""
    We retrieved **252 trading days** (1 year) of data. 
    -   Current Price: **${current_price:.2f}**
    -   If the blue line (50 SMA) is above the orange line (200 SMA), it indicates a "Golden Cross" (Bullish).
    -   If the price is far above the averages, the stock might be "extended" or "overbought".
    """)
    
    # Collapsible Data Table
    with st.expander("📊 View Complete 12-Month Historical Data Table", expanded=False):
        st.markdown("#### Complete NVDA Price Data (Last 12 Months)")
        
        # Prepare data for display
        display_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        display_df.index = display_df.index.strftime('%Y-%m-%d')
        display_df = display_df.sort_index(ascending=False)  # Most recent first
        
        # Format the data nicely
        st.dataframe(
            display_df.style.format({
                'Open': '${:.2f}',
                'High': '${:.2f}',
                'Low': '${:.2f}',
                'Close': '${:.2f}',
                'Volume': '{:,.0f}'
            }),
            use_container_width=True,
            height=400
        )
        
        st.caption(f"📅 Showing {len(display_df)} trading days from {display_df.index[-1]} to {display_df.index[0]}")


# --- TAB 3: VOLATILITY ---
with tabs[2]:
    st.header("3. Historical Volatility Analysis")
    
    st.markdown("""
    <div class='edu-box'>
        <span class='theory-header'>📉 Theory: Volatility (Sigma σ)</span>
        <p>Volatility is the <strong>most critical input</strong> in option pricing. It measures the "chaos" or magnitude of price swings.</p>
        <p>We don't know the future volatility, so we often look at the past ("Historical Volatility") as a guide. We assume stock returns follow a <strong>Log-Normal Distribution</strong> (Bell Curve).</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 🔢 The Calculation")
        st.latex(r"r_t = \ln(\frac{P_t}{P_{t-1}})")
        st.latex(r"\sigma = \text{StDev}(r_t) \times \sqrt{252}")
        
        st.markdown(f"""
        <div class='concept-box'>
            <span class='concept-title'>Result</span>
            <div style='font-size: 2rem; font-weight: 800; color: {C_BLUE};'>{hist_vol*100:.2f}%</div>
            <span>Annualized Volatility</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🧐 What does this number mean?")
        st.write(f"""
        A volatility of **{hist_vol*100:.0f}%** implies that, statistically speaking, there is a ~68% chance (1 standard deviation) that NVDA's price 1 year from now will be within ±{hist_vol*100:.0f}% of today's price.
        
        -   **Low Volatility (<20%)**: Stock is calm, like a utility company. Options are **Cheap**.
        -   **High Volatility (>50%)**: Stock is wild, like a crypto or biotech. Options are **Expensive**.
        
        NVDA is a high-growth tech stock, so we expect higher volatility than the S&P 500.
        """)
        
    # Distribution Plot
    fig_hist = px.histogram(df, x="Log_Ret", nbins=60, title="Distribution of Daily Returns (The 'Bell Curve')", 
                            color_discrete_sequence=[C_BLUE], opacity=0.7)
    
    # Add Bell Curve Overlay
    x_range = np.linspace(min(df['Log_Ret']), max(df['Log_Ret']), 100)
    # Fit normal dist
    clean_ret = df['Log_Ret'].dropna()
    mu, std = si.norm.fit(clean_ret)
    pdf = si.norm.pdf(x_range, mu, std)
    # Scale pdf to match histogram count
    # This is rough visualization scaling
    scale_factor = len(df) * (max(df['Log_Ret']) - min(df['Log_Ret'])) / 60
    
    fig_hist.update_layout(template="plotly_white", xaxis_title="Daily Log Return", yaxis_title="Frequency of Days")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption("The chart above shows how often NVDA had a return of X%. Noticed how it centers around 0%? Most days are quiet, but the 'tails' show the extreme days.")

# --- TAB 4: PRICING ---
with tabs[3]:
    st.header("4. Black-Scholes Pricing Model")
    
    st.markdown("""
    <div class='edu-box'>
        <span class='theory-header'>🧮 Theory: The Black-Scholes-Merton Formula</span>
        <p>This Nobel-prize winning formula calculates the theoretical fair value of an option. It works by creating a <strong>risk-free portfolio</strong> that replicates the option's payout.</p>
        <p><strong>The Call Option Price Formula:</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display formulas using st.latex for proper rendering
    st.latex(r"C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)")
    
    st.markdown("""
    **Where:**
    - **S · N(d₁)** = Expected Benefit of buying the stock
    - **K · e⁻ʳᵀ · N(d₂)** = Expected Cost (discounted strike price)
    """)
    
    # Inputs
    st.markdown("### 1. Model Inputs")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spot Price (S)", f"${ASSUMED_S:.2f}", help="Current NVDA Price")
    c2.metric("Strike Price (K)", f"${ASSUMED_K:.2f}", help="Price we agree to buy at")
    c3.metric("Time (T)", f"{ASSUMED_T} Years", help="3 Months / 12")
    c4.metric("Risk-Free (r)", f"{ASSUMED_R*100}%", help="US Treasury Bill Rate")
    c5.metric("Volatility (σ)", f"{ASSUMED_SIGMA*100:.1f}%", help="Annualized Std Dev")
    
    # Calculate
    model = BSModel(ASSUMED_S, ASSUMED_K, ASSUMED_T, ASSUMED_R, ASSUMED_SIGMA, "call")
    price = model.price()
    
    st.markdown("### 2. The Result")
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {C_BLUE}, {C_DARK}); padding: 30px; border-radius: 15px; text-align: center; color: white; margin: 20px 0;'>
        <h2 style='color: white !important; border: none; margin: 0; padding: 0;'>Fair Value: ${price:.2f}</h2>
        <p style='color: #E5E7EB; margin-top: 10px; font-size: 1.1rem;'>This is the theoretical price you should pay for this contract.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 3. Visualizing Profit & Loss (PnL)")
    st.write("If you buy this option for the calculated price, when do you make money?")
    
    # Interactive Payoff Diagram
    s_range = np.linspace(ASSUMED_S * 0.7, ASSUMED_S * 1.3, 100)
    # PnL = Max(S - K, 0) - Premium
    payoffs = [max(s - ASSUMED_K, 0) - price for s in s_range]
    
    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(x=s_range, y=payoffs, mode='lines', name='Net PnL', fill='tozeroy', 
                                    line=dict(color=C_GREEN, width=3)))
    
    # Add breakeven line
    breakeven = ASSUMED_K + price
    
    fig_payoff.add_vline(x=ASSUMED_S, line_dash="dot", annotation_text="Spot", line_color=C_BLUE)
    fig_payoff.add_vline(x=ASSUMED_K, line_dash="solid", annotation_text="Strike", line_color=C_DARK)
    fig_payoff.add_vline(x=breakeven, line_dash="dash", annotation_text=f"Break-even (${breakeven:.0f})", line_color=C_RED)
    
    fig_payoff.add_shape(type="rect",
        xref="x", yref="paper",
        x0=min(s_range), y0=0, x1=max(s_range), y1=0,
        line=dict(color="black", width=1),
    )
    
    fig_payoff.update_layout(template="plotly_white", title="Option Profit/Loss at Expiry", 
                             xaxis_title="NVDA Price at Expiry ($)", yaxis_title="Net Profit ($)")
    st.plotly_chart(fig_payoff, use_container_width=True)
    
    st.info(f"💡 You only start making money if NVDA rises above **${breakeven:.2f}** (Strike + Premium). Below that, you lose money.")

# --- TAB 5: GREEKS ---
with tabs[4]:
    st.header("5. The Greeks (Risk Sensitivities)")
    
    st.markdown("""
    <div class='edu-box'>
        <span class='theory-header'>📐 Theory: Managing the Machine</span>
        <p>"Greeks" are the dashboard derivatives that tell us how the option price will change when market variables move.</p>
        <p>Think of the Option Price as a car. The Greeks tell us how fast it goes, how much it vibrates, and how much fuel it burns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    delta = model.delta()
    theta = model.theta()
    vega = model.vega()
    
    # Greek Cards
    g1, g2, g3 = st.columns(3)
    
    with g1:
        st.markdown(f"""
        <div class='concept-box' style='border-color: {C_BLUE}; background-color: #EFF6FF;'>
            <span class='concept-title' style='color: {C_BLUE};'>Δ Delta (Speed)</span>
            <div style='font-size: 1.8rem; font-weight: 700; color: {C_BLUE};'>{delta:.3f}</div>
            <p style='font-size: 0.9rem; margin: 10px 0;'>For every $1 NVDA moves UP, the option gains approximately <strong>${delta:.2f}</strong>.</p>
            <hr style='border: none; border-top: 1px solid #DBEAFE; margin: 10px 0;'>
            <p style='font-size: 0.85rem; margin: 10px 0;'><i>Interpretation:</i> Also represents the approximate <strong>Probability</strong> the option finishes In-The-Money (~<strong>{delta*100:.0f}%</strong> chance).</p>
        </div>
        """, unsafe_allow_html=True)
        
    with g2:
        st.markdown(f"""
        <div class='concept-box' style='border-color: {C_RED}; background-color: #FEF2F2;'>
            <span class='concept-title' style='color: {C_RED};'>Θ Theta (Time Decay)</span>
            <div style='font-size: 1.8rem; font-weight: 700; color: {C_RED};'>${theta:.3f}</div>
            <p style='font-size: 0.9rem; margin: 10px 0;'>Every single day that passes, this option loses <strong>${abs(theta):.2f}</strong> in value, assuming price stays flat.</p>
            <hr style='border: none; border-top: 1px solid #FECACA; margin: 10px 0;'>
            <p style='font-size: 0.85rem; margin: 10px 0;'><i>Interpretation:</i> Options are "wasting assets". Time is your enemy as a buyer.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with g3:
        st.markdown(f"""
        <div class='concept-box' style='border-color: {C_AMBER}; background-color: #FFFBEB;'>
            <span class='concept-title' style='color: {C_AMBER};'>ν Vega (Volatility)</span>
            <div style='font-size: 1.8rem; font-weight: 700; color: {C_AMBER};'>${vega:.3f}</div>
            <p style='font-size: 0.9rem; margin: 10px 0;'>If market panic (Volatility) increases by 1%, the option value GAINS <strong>${vega:.2f}</strong>.</p>
            <hr style='border: none; border-top: 1px solid #FDE68A; margin: 10px 0;'>
            <p style='font-size: 0.85rem; margin: 10px 0;'><i>Interpretation:</i> Long options are "Long Volatility". You benefit from chaos.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🧊 3D Volatility Pricing Surface")
    st.write("This interactive chart shows how **Price (Height)** depends on both **Stock Price** and **Volatility** simultaneously.")
    
    spot_range = np.linspace(ASSUMED_S * 0.8, ASSUMED_S * 1.2, 25)
    vol_range = np.linspace(0.1, 0.8, 25)
    s_mesh, v_mesh = np.meshgrid(spot_range, vol_range)
    z_mesh = np.zeros_like(s_mesh)
    
    for i in range(len(vol_range)):
        for j in range(len(spot_range)):
            z_mesh[i,j] = BSModel(s_mesh[i,j], ASSUMED_K, ASSUMED_T, ASSUMED_R, v_mesh[i,j]).price()
            
    fig_3d = go.Figure(data=[go.Surface(z=z_mesh, x=s_mesh, y=v_mesh, colorscale='Viridis')])
    fig_3d.update_layout(title="", scene=dict(xaxis_title='Spot ($)', yaxis_title='Vol (%)', zaxis_title='Price ($)'), 
                         height=600, margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig_3d, use_container_width=True)

# --- TAB 6: RISK ---
with tabs[5]:
    st.header("6. Risk Management: Delta Hedging")
    
    st.markdown("""
    <div class='edu-box'>
        <span class='theory-header'>🛡️ Theory: How Market Makers survive</span>
        <p>Investment banks sell millions of options. If they just held them, they would be gambling. Instead, they use <strong>Delta Hedging</strong> to become "Market Neutral".</p>
        <p>If you own an option with Delta 0.60, it acts like 60 shares of stock. To cancel this out, you must <strong>Short Sell 60 shares</strong> of the real stock.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎲 Monte Carlo Simulation")
    st.write(f"We simulated **1,000 possible futures** for NVDA over the next 3 months to see the probability of profit.")
    
    np.random.seed(42)
    dt = ASSUMED_T
    sims = 1000
    Z = np.random.normal(0, 1, sims)
    S_T = ASSUMED_S * np.exp((ASSUMED_R - 0.5*ASSUMED_SIGMA**2)*dt + ASSUMED_SIGMA*np.sqrt(dt)*Z)
    
    # Probability of Profit
    ITM_prob = np.sum(S_T > ASSUMED_K) / sims
    
    fig_mc = px.histogram(S_T, nbins=50, title="Projected NVDA Price Distribution at Expiry", 
                          color_discrete_sequence=[C_AMBER])
    fig_mc.add_vline(x=ASSUMED_K, line_color=C_DARK, line_dash="dash", annotation_text="Strike Price")
    fig_mc.add_vline(x=ASSUMED_S, line_color=C_BLUE, line_dash="dot", annotation_text="Today")
    fig_mc.update_layout(template="plotly_white", xaxis_title="Price ($)", yaxis_title="Frequency")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig_mc, use_container_width=True)
    with col2:
        st.markdown("#### Simulation Results")
        st.metric("Simulations", sims)
        st.metric("Mean Final Price", f"${np.mean(S_T):.2f}")
        st.metric("Prob. of Profit", f"{ITM_prob*100:.1f}%", delta_color="normal" if ITM_prob > 0.5 else "inverse")
        st.write("This probability closely matches our **Delta** calculation!")

# --- TAB 7: SENSITIVITY ---
with tabs[6]:
    st.header("7. Sensitivity Analysis (Stress Testing)")
    st.write("Financial markets are dynamic. We use 'Ceteris Paribus' (all else equal) analysis to see how one changing variable affects our wealth.")
    
    # Table Construction
    scenarios = []
    scenarios.append({"Scenario": "Base Case", "Vol Input": f"{ASSUMED_SIGMA*100:.1f}%", "Time Input": "3.0 Mo", "Fair Value": price})
    
    p_vol_up = BSModel(ASSUMED_S, ASSUMED_K, ASSUMED_T, ASSUMED_R, ASSUMED_SIGMA+0.10).price()
    scenarios.append({"Scenario": "😱 Panic (Vol +10%)", "Vol Input": f"{(ASSUMED_SIGMA+0.10)*100:.1f}%", "Time Input": "3.0 Mo", "Fair Value": p_vol_up})
    
    p_time_down = BSModel(ASSUMED_S, ASSUMED_K, 0.1, ASSUMED_R, ASSUMED_SIGMA).price()
    scenarios.append({"Scenario": "⏳ Time Decay (1.2 Mo left)", "Vol Input": f"{ASSUMED_SIGMA*100:.1f}%", "Time Input": "1.2 Mo", "Fair Value": p_time_down})
    
    p_crash = BSModel(ASSUMED_S*0.9, ASSUMED_K, ASSUMED_T, ASSUMED_R, ASSUMED_SIGMA+0.20).price()
    scenarios.append({"Scenario": "📉 Crash (Price -10%, Vol +20%)", "Vol Input": f"{(ASSUMED_SIGMA+0.20)*100:.1f}%", "Time Input": "3.0 Mo", "Fair Value": p_crash})

    df_sens = pd.DataFrame(scenarios)
    
    st.table(df_sens.style.format({"Fair Value": "\${:.2f}"}).applymap(lambda v: 'color: red;' if 'Time' in str(v) else ('color: green;' if 'Panic' in str(v) else None), subset=["Scenario"]))
    
    st.markdown("### 💡 Key Insight")
    st.info("Notice the 'Crash' scenario. Even though the stock price fell 10% (bad for call), the Volatility spiked 20% (good for call). The option might not lose as much value as you think because Vega offsets Delta. This is why options are complex!")

# --- TAB 8: CALL vs PUT ---
with tabs[7]:
    st.header("8. Call vs Put Comparison")
    
    st.markdown("""
    <div class='edu-box'>
        <span class='theory-header'>⚖️ Theory: Put-Call Parity</span>
        <p>A <strong>Call</strong> profits when markets rise. A <strong>Put</strong> profits when markets fall.</p>
        <p>They are mathematically linked. If you know the price of one, you can determine the price of the other using the risk-free rate and the stock price.</p>
    </div>
    """, unsafe_allow_html=True)
    
    p_put = BSModel(ASSUMED_S, ASSUMED_K, ASSUMED_T, ASSUMED_R, ASSUMED_SIGMA, "put").price()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 🐂 CALL Option")
        st.write("Rights to **BUY** NVDA.")
        st.metric("Price", f"${price:.2f}")
        st.success("Strategy: Bullish Speculation")
        
        # Mini chart
        x = np.linspace(ASSUMED_S*0.8, ASSUMED_S*1.2, 50)
        y = [max(price_ - ASSUMED_K, 0) for price_ in x]
        st.line_chart(pd.DataFrame({'Price': y}, index=x), height=150)
        
    with col2:
        st.markdown(f"### 🐻 PUT Option")
        st.write("Rights to **SELL** NVDA.")
        st.metric("Price", f"${p_put:.2f}")
        st.error("Strategy: Bearish / Insurance")
        
        # Mini chart
        y_put = [max(ASSUMED_K - price_, 0) for price_ in x]
        st.line_chart(pd.DataFrame({'Price': y_put}, index=x), height=150)

# --- TAB 9: LIMITATIONS ---
with tabs[8]:
    st.header("9. Model Limitations (Real World Reality Check)")
    
    st.markdown("""
    <div class='risk-box'>
        <h3 style='color: #991B1B !important; margin-top: 0; margin-bottom: 15px;'>⚠️ WARNING: The Map is Not the Territory</h3>
        <p style='font-size: 1.05rem; line-height: 1.6;'>The Black-Scholes model relies on <strong>assumptions that do not always hold true</strong> in the real world. Understanding these limitations is crucial for risk management.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    | Assumption | Reality | Risk Impact |
    | :--- | :--- | :--- |
    | **Constant Volatility** | Volatility changes dynamically and spikes during crashes. | Model **underprices tail risk** (Result: "Volatility Smile"). |
    | **No Jumps** | Stocks can gap up/down overnight (earnings, news). | **Gamma risk** is underestimated. You cannot hedge a gap. |
    | **Normal Distribution** | Returns have "Fat Tails" (Kurtosis). | Extreme events (Black Swans) happen far more often than the model predicts. |
    | **Frictionless** | Trading has costs (spreads, fees). | **Transaction costs** can eat up all theoretical profits from hedging. |
    """)

# --- TAB 10: INTERACTIVE SIMULATOR ---
with tabs[9]:
    st.header("🎛️ Interactive Educational Sandbox")
    st.markdown("""
    <div class='edu-box'>
        <span class='theory-header'>🧪 Experiment & Learn</span>
        <p>Change the inputs below to see how the mathematical model responds. Try these experiments:</p>
        <ol>
            <li><strong>Time Decay:</strong> Set Time to 0.01 (Expiring tomorrow). Watch the Price collapse.</li>
            <li><strong>Deep ITM:</strong> Lower Strike to $200. Watch Delta go to 1.0 (The option becomes the stock).</li>
            <li><strong>Panic Mode:</strong> Increase Volatility to 100%. Watch the Option Price explode.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_viz = st.columns([1, 2])
    
    with col_input:
        st.subheader("⚙️ Inputs")
        sim_S = st.number_input("Stock Price ($)", value=float(ASSUMED_S), step=1.0, key="sim_s")
        sim_K = st.number_input("Strike Price ($)", value=float(ASSUMED_K), step=1.0, key="sim_k")
        sim_T = st.slider("Time to Expiry (Years)", min_value=0.01, max_value=2.0, value=float(ASSUMED_T), step=0.01, key="sim_t")
        sim_r = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=float(ASSUMED_R*100), step=0.5, key="sim_r") / 100.0
        sim_sigma = st.slider("Volatility (%)", min_value=10.0, max_value=150.0, value=float(ASSUMED_SIGMA*100), step=1.0, key="sim_sigma") / 100.0
    
    # User Model
    u_model = BSModel(sim_S, sim_K, sim_T, sim_r, sim_sigma, "call")
    u_price = u_model.price()
    u_delta = u_model.delta()
    u_theta = u_model.theta()
    u_vega = u_model.vega()
    
    with col_viz:
        st.subheader("📊 Dynamic Output")
        
        # Large Output Metrics
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {C_BLUE}, {C_GREEN}); padding: 25px; border-radius: 15px; text-align: center; color: white; margin-bottom: 20px;'>
            <h3 style='color: white !important; margin: 0; padding: 0;'>Option Price: \\${u_price:.2f}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Δ Delta", f"{u_delta:.3f}", help="Sensitivity to stock price changes")
        c2.metric("Θ Theta", f"${u_theta:.3f}", delta=f"${u_theta:.3f}/day", delta_color="inverse", help="Daily time decay")
        c3.metric("ν Vega", f"${u_vega:.3f}", help="Sensitivity to volatility changes")
        c4.metric("Prob ITM", f"{u_delta*100:.1f}%", help="Probability of finishing in-the-money")
        
        # Dynamic Delta Plot
        st.markdown("---")
        st.markdown("#### Delta Curve Visualization")
        s_sim_range = np.linspace(sim_S * 0.5, sim_S * 1.5, 100)
        deltas = [BSModel(s, sim_K, sim_T, sim_r, sim_sigma).delta() for s in s_sim_range]
        
        fig_sim = px.line(x=s_sim_range, y=deltas, title="Delta Curve (The 'S' Curve)", 
                          labels={'x':'Stock Price ($)', 'y':'Delta (Probability)'})
        fig_sim.add_vline(x=sim_S, line_dash="dash", annotation_text="Current Price", line_color=C_AMBER)
        fig_sim.add_vline(x=sim_K, line_dash="solid", annotation_text="Strike Price", line_color=C_RED)
        fig_sim.update_traces(line_color=C_BLUE, line_width=4)
        fig_sim.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig_sim, use_container_width=True)
        
        st.info(f"💡 **Interpretation:** With current inputs, if NVDA moves up by $1, your option gains approximately ${u_delta:.2f}. You lose ${abs(u_theta):.2f} every day to time decay.")
