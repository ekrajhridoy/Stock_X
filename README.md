# Stock_X 📈
# An Integrated Stock, Gold & Financial News Analytics System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 🚀 Overview

**Stock_X** is an AI-powered financial intelligence platform that integrates **stock market data**, **gold price information**, and **financial news analytics** into a single interactive application.

Built using **Python** and **Streamlit**, the platform helps investors, students, and researchers explore financial markets through advanced analytics, machine learning, forecasting, sentiment analysis, portfolio optimization, and explainable AI.

The system uses historical **NIFTY-50 stock market data**, gold prices, and financial news headlines to generate investor-friendly insights and predictions.

---

## 🎯 Objectives

Stock_X aims to:

* Centralize stock, gold, and financial news data.
* Provide interpretable AI-driven financial insights.
* Deliver machine learning and forecasting capabilities.
* Support portfolio construction and investment decisions.
* Offer explainable predictions using SHAP and LIME.
* Maintain reproducibility through modular architecture.
* Create an educational and investor-friendly analytics environment.

---

## ✨ Key Features

### 📊 Executive Dashboard

* Interactive KPI cards
* Market trend visualization
* Candlestick charts
* Volume analysis
* Correlation heatmaps
* Real-time filtering

---

### 🔍 Exploratory Data Analysis (EDA)

Provides **100+ investor-focused analytical questions** including:

#### Data Understanding

* Dataset Overview
* Missing Value Analysis
* Industry Distribution

#### Univariate Analysis

* Stock Price Distribution
* Volume Distribution
* Gold Price Trends

#### Bivariate Analysis

* Stock vs Gold Relationship
* Volume vs Price Movement

#### Multivariate Analysis

* Correlation Analysis
* Feature Relationships

#### Time Series Analysis

* Trend Detection
* Moving Average Analysis
* Seasonal Patterns

#### Risk Analysis

* Volatility Analysis
* Return Distribution

Each question is visualized using interactive Plotly charts and accompanied by automatically generated insights.

---

## 🤖 Machine Learning Modules

### 1️⃣ Stock Direction Prediction

**Question:** Will the stock price go up or down tomorrow?

Models:

* XGBoost
* Random Forest
* LightGBM (Optional)

Output:

* BUY / SELL Signal
* Probability Score
* Model Accuracy

---

### 2️⃣ Feature Importance Analysis

**Question:** Which factors drive stock returns?

Techniques:

* Random Forest Feature Importance
* SHAP Explanations

Outputs:

* Importance Rankings
* Interactive Visualizations

---

### 3️⃣ Volatility Clustering

**Question:** Which stocks are riskier?

Techniques:

* K-Means Clustering
* Random Forest Validation

Outputs:

* Risk Tiers
* Cluster Analysis

---

### 4️⃣ Market Crash Detection

**Question:** Is today an abnormal trading day?

Technique:

* Isolation Forest

Outputs:

* Anomaly Detection
* Crash Probability Indicators

---

## 📈 Forecasting & Time-Series Analytics

### Future Forecasting

Predict future stock prices using:

* LSTM
* GRU
* Ridge Regression Fallback

Features:

* User-selectable forecasting horizon
* Confidence intervals
* Multi-step forecasting

---

### Forecast Comparison

Compare forecasting models:

* ARIMA
* Prophet
* LSTM
* GRU

Metrics:

* RMSE
* MAE
* MAPE

---

## 📰 News Intelligence

### News Investor Assistant

Analyze financial news sentiment using:

* Financial Lexicon-Based Sentiment Analysis
* FinBERT (Optional)

Outputs:

* News Health Score
* Impact Meter
* Positive/Negative Sentiment Analysis
* Historical News Impact

---

### News Sensitivity Analysis

Identify which stocks react most strongly to news events.

Features:

* Company-wise sensitivity ranking
* Comparative analysis

---

### Headline Impact Prediction

Predict whether a headline will cause:

* 📈 Upward Movement
* 📉 Downward Movement
* ➖ Neutral Impact

Models:

* Sentiment Analysis
* Technical Indicators
* XGBoost

---

## 🧠 Explainable AI

### SHAP Explanations

Understand why a prediction was made.

Features:

* Global Importance
* Local Importance
* Interactive SHAP Plots

---

### LIME Explanations

Generate human-readable prediction explanations.

---

## 🔄 Historical Analogues

Find similar historical stock patterns using:

* Dynamic Time Warping (DTW)
* Euclidean Similarity

Outputs:

* Similarity Score
* Bullish Probability
* Bearish Probability
* Outcome Distribution
* Average Return

---

## 🎮 Reinforcement Learning Trading Assistant

Provides intelligent trading recommendations.

Outputs:

* Buy / Hold / Sell Decision
* Decision Log
* Performance Scorecard

Algorithms:

* Q-Learning
* Deep Q Network (DQN)

---

## 💼 Smart Portfolio Construction

### Portfolio Optimization

Create optimized investment portfolios using:

* Sharpe Ratio Weighting
* Composite Quality Scores

Features:

* Risk Profiling
* Diversification Analysis
* Investment Allocation Suggestions

---

## 📝 Investment Thesis Generator

Generate evidence-based investment recommendations.

Outputs:

* Buy
* Hold
* Avoid

Based on:

* Technical Indicators
* Forecasts
* News Sentiment
* Risk Analysis

---

## 🛠️ Technology Stack

### Frontend

* Streamlit
* Plotly
* Streamlit Option Menu
* Streamlit AgGrid
* Custom Glassmorphism UI

### Backend

* Python
* Pandas
* NumPy
* Scikit-Learn
* StatsModels
* SciPy

### Optional Advanced Libraries

* XGBoost
* LightGBM
* CatBoost
* SHAP
* LIME
* TensorFlow
* Prophet
* Transformers
* PyTorch
* Stable-Baselines3
* TSLearn

---

## 📂 Project Structure

```text
Stock_X/
│
├── main.py
│
├── data/
│   └── cleaned_merged_data.csv
│
├── modules/
│   ├── eda.py
│   ├── stock_direction_prediction.py
│   ├── feature_importance_analysis.py
│   ├── future_forecasting.py
│   ├── volatility_clustering.py
│   ├── crash_detection.py
│   ├── news_investor_assistant.py
│   ├── news_sensitivity.py
│   ├── historical_analogues.py
│   ├── explainable_ai.py
│   ├── smart_portfolio.py
│   ├── investment_thesis.py
│   └── rl_trading_assistant.py
│
└── utils/
```

---

## 📊 Dataset Information

Dataset Includes:

* Historical Stock Data
* Gold Prices
* Financial News Headlines

Coverage:

* 49 Stock Symbols
* 13 Industries
* ~120,000 Records
* NIFTY-50 Market

Engineered Features:

* RSI
* MACD
* Momentum
* Volatility
* Daily Returns
* Log Returns
* SMA
* EMA
* Lag Features
* Next-Day Direction Labels

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Stock_X.git
cd Stock_X
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run main.py
```

The application will open automatically in your browser.

---

## 📸 Application Modules

| Module                | Purpose                 |
| --------------------- | ----------------------- |
| Dashboard             | Market Overview         |
| EDA                   | Data Exploration        |
| Direction Prediction  | Buy/Sell Signal         |
| Feature Importance    | Driver Analysis         |
| Forecasting           | Future Prices           |
| Volatility Clustering | Risk Categorization     |
| Crash Detection       | Anomaly Detection       |
| News Assistant        | Sentiment Analysis      |
| Historical Analogues  | Pattern Matching        |
| Explainable AI        | Prediction Explanation  |
| RL Assistant          | Trading Recommendations |
| Smart Portfolio       | Portfolio Optimization  |
| Investment Thesis     | Decision Support        |

---

## 🔮 Future Enhancements

* Live Market Data Integration
* Real-Time News Streaming
* Mobile Application
* Cloud Deployment
* User Authentication
* Watchlists
* Email Alerts
* PDF Investor Reports
* Natural Language Querying
* Managed Database Migration
* Multi-Stock Backtesting

---

## 👨‍💻 Authors

### Ariful Ekraj Hridoy

ID: 0152410013

### Rubaya Tabassum

ID: 0152410016




⭐ If you found this project useful, please consider giving it a star on GitHub!
