# NVDA Option Pricing & Risk Analysis

A comprehensive financial derivatives analysis application built with Python and Streamlit, demonstrating Black-Scholes option pricing, Greeks calculations, and risk assessment for NVIDIA (NVDA) stock options.

## 🎯 Features

### Educational & Interactive
- **10 Comprehensive Tabs** covering complete option pricing theory and practice
- **Real-time Market Data** from Yahoo Finance (12 months historical NVDA data)
- **Interactive Simulator** to experiment with option pricing parameters
- **Beautiful Visualizations** using Plotly (3D surfaces, candlestick charts, Monte Carlo simulations)

### Core Functionality
1. **Overview** - Introduction to derivatives and options
2. **Market Data** - Live NVDA data with technical analysis (candlesticks, moving averages)
3. **Volatility Analysis** - Historical volatility calculation and distribution analysis
4. **Black-Scholes Pricing** - Complete option valuation with profit/loss visualization
5. **Greeks** - Delta, Theta, Vega calculations with 3D pricing surface
6. **Risk & Hedging** - Monte Carlo simulation and delta hedging strategies
7. **Sensitivity Analysis** - Scenario testing and stress analysis
8. **Call vs Put** - Comparison with put-call parity
9. **Model Limitations** - Real-world constraints and assumptions
10. **Interactive Simulator** - Real-time parameter adjustment with live outputs

## 📊 Application Preview

The app features:
- Clean, professional UI with color-coded sections
- Horizontal scrolling tabs for easy navigation
- Collapsible data tables
- Real-time calculations
- Educational content with formulas and explanations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/aishwanth-dev/NVDA-Option-Pricing-Risk-Analysis.git
cd NVDA-Option-Pricing-Risk-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 Dependencies

- `streamlit` - Web application framework
- `yfinance` - Real-time stock data from Yahoo Finance
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scipy` - Scientific computing (statistical distributions)
- `plotly` - Interactive visualizations
- `matplotlib` - Additional plotting capabilities

## 🌐 Deploy to Streamlit Cloud

### Steps:
1. Push this code to your GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select:
   - Repository: `aishwanth-dev/NVDA-Option-Pricing-Risk-Analysis`
   - Branch: `main`
   - Main file path: `app.py`
6. Click "Deploy"

Your app will be live at: `https://[your-app-name].streamlit.app`

## 🧮 Technical Details

### Black-Scholes Model
The application implements the complete Black-Scholes-Merton formula for European options:

**Call Option Price:**
```
C = S·N(d₁) - K·e⁻ʳᵀ·N(d₂)
```

### Greeks Calculations
- **Delta (Δ)**: Rate of change of option price with respect to stock price
- **Theta (Θ)**: Time decay of option value
- **Vega (ν)**: Sensitivity to volatility changes

### Risk Analysis
- Monte Carlo simulation with 1,000 price paths
- Historical volatility using log returns
- Delta hedging strategies

## 📚 Educational Value

This application is designed for:
- **Finance Students** learning option pricing theory
- **Quantitative Analysts** practicing risk assessment
- **Traders** understanding option Greeks and sensitivities
- **Educators** teaching derivatives and financial engineering

## 🎨 UI Features

- **Responsive Design** with horizontal tab scrolling
- **Color-Coded Sections** for easy navigation
- **Interactive Charts** with zoom and pan capabilities
- **Professional Styling** with gradients and shadows
- **Clean Typography** using modern fonts

## ⚠️ Disclaimer

This application is for **educational purposes only**. The Black-Scholes model makes several assumptions that may not hold in real markets. Always consult with qualified financial professionals before making investment decisions.

## 📄 License

MIT License - feel free to use this for educational purposes!

## 👨‍💻 Author

**Aishwanth**
- GitHub: [@aishwanth-dev](https://github.com/aishwanth-dev)

## 🙏 Acknowledgments

- Yahoo Finance for real-time market data
- Streamlit for the amazing web framework
- Plotly for beautiful interactive visualizations

---

**Built with ❤️ using Python, Streamlit, and Financial Mathematics**
