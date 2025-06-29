# 📊 Profitable trading strategy: SOXL Exploratory Data Analysis

---

## 🧠 Research Question

**Can we develop a profitable trading strategy using the SuperTrend indicator with advanced risk management and choppiness filters to avoid unfavorable market conditions?**

---

## 📚 Dataset Overview

This analysis uses historical price data for **SOXL** (a 3x Leveraged ETF) at **10-minute intervals**, including several years of price data (6.43 MB)


- **OHLCV**: Open, High, Low, Close, Volume
- **SuperTrend** indicators
- 15 engineered features derived from price, volume, and technical signals

Jupyter Notebook: Run it in Google Collab
-  https://github.com/cfphpflex/SuperTrend-Trading-Strategy---Exploratory-Data-Analysis-EDA-/blob/main/SuperTrend_Trading_EDA.ipynb

---

## 🧼 Data Quality and Cleaning

- ✅ Successfully loaded and cleaned SOXL 10-minute price data  (years worth 6.43 MB)
- ✅ Removed duplicates and handled missing values  
- ✅ Detected and addressed outliers  
- ✅ Engineered 15 technical features for in-depth analysis  

---

## 🔍 Exploratory Data Analysis (EDA)

- 📈 **Price Analysis**: SOXL displays high volatility, with price changes ranging from **-X% to +X%**
- 📊 **Volume Analysis**: Volume fluctuates significantly throughout the trading day
- ⏰ **Time Patterns**: Clear intraday trends in price, volatility, and volume
- 🔗 **Correlations**: Strong links observed between price movement and technical indicators

---

## 🤖 Machine Learning Baseline

- **Model**: Random Forest  
- **Performance**: Achieved **X% accuracy** vs **X% baseline**
- 🔝 **Top Predictive Features**:
  - Price change
  - Volatility
  - Moving averages  
- 📊 **Classification**: Model shows balanced up/down prediction performance

---

## 💡 Trading Strategy Implications

- ⌛ Time-based filtering may improve trade performance
- 📈 Volatility-based features are strong predictors of direction
- 📊 Volume provides key insights into trade timing
- 📐 Technical indicators demonstrate high predictive value

---

## 🛠️ Next Steps for Module 24

- 🚀 **Advanced Models**: Explore XGBoost, Neural Networks, and ensemble techniques  
- 🧪 **Feature Selection**: Refine based on feature importance and SHAP values  
- ⚙️ **Hyperparameter Tuning**: Optimize models for improved accuracy and robustness  
- 📉 **Strategy Integration**: Combine ML predictions with SuperTrend signals  
- 🛡️ **Risk Management**: Dynamic position sizing based on model confidence  

---

## ⚠️ Limitations and Considerations

- 📉 **Market Regimes**: Performance may vary under different macroeconomic conditions  
- 🧠 **Overfitting**: Emphasize validation with out-of-sample and cross-validation methods  
- 💸 **Transaction Costs**: Account for slippage, spreads, and fees  
- ⚖️ **Risk Controls**: Always pair ML predictions with sound risk management frameworks  

---

## ✅ Summary

This initial analysis lays a strong foundation for building a **SuperTrend-enhanced trading strategy** powered by **machine learning**. The next module will focus on optimizing predictive models and integrating them into a real-world trading framework.




python build_proper_cache.py -s CYN -t 1Min -d 365
python build_proper_cache.py -s CYN -t 5Min -d 365
python build_10min_from_1min.py --symbol CYN --days 365


python build_proper_cache.py -s SOXL -t 5Min -d 365
python build_proper_cache.py -s SOXL -t 10Min -d 365
python build_30min_from_1min.py --symbol NVDA --days 365

python build_proper_cache.py -s SMCI -t 1Min -d 365
python build_proper_cache.py -s SMCI -t 5Min -d 365
python build_10min_from_1min.py --symbol SMCI --days 365


python build_proper_cache.py -s SOXL -t 10Min -d 365
python build_proper_cache.py -s NVDA -t 5Min -d 365
python build_proper_cache.py -s TSLA -t 5Min -d 365
python build_proper_cache.py -s CRCL -t 1Min -d 150
python build_proper_cache.py -s CRCL -t 5Min -d 150
python build_proper_cache.py -s TQQQ -t 5Min -d 365

python build_proper_cache.py -s META -t 5Min -d 365
python build_10min_from_1min.py --symbol SOXL --days 1850      5 years


 
python build_5min_from_1min.py --symbol SOXL


-- NEW 062425 analyze_trades_supertrend
python analyze_trades_supertrend.py


python verify_system_status.py

BACKTEST
# SOXL

=====FINAL WIN TESTS=====>>>>>>  WINNER------WINNER-------WINNER  10Min---WINNER-WINNER----WINNER---WINNER <<<<<<==========

 
# SOXL OPTIMIZE & NON-OPTIZE

#optimize stop loss & risk_per_trade

python analyze_trades_supertrend.py --symbol SOXL --timeframe 10Min --initial_capital 1000 --start_date 2020-01-01 --end_date 2025-06-25 --enhanced --compounded --optimize  --risk_per_trade 0.01 --stop_loss_range 1 12 --optimization_iterations 100

python analyze_trades_supertrend.py --symbol SOXL --timeframe 10Min --start_date 2022-06-28 --end_date 2025-06-28

# NON-Optimzied  1 yr
python analyze_trades_supertrend.py --symbol SOXL --timeframe 10Min --initial_capital 1000 --start_date 2022-06-28 --end_date 2025-06-28 --enhanced --skip_optimization --compounded --risk_per_trade 0.01 --stop_loss_pct 1.0

# TQQQ
python analyze_trades_supertrend.py --symbol TQQQ --timeframe 10Min --initial_capital 1000 --start_date 2024-06-01 --end_date 2025-06-25 --enhanced --skip_optimization --compounded
 
# CRCL
 
python analyze_trades_supertrend.py --symbol CRCL --timeframe 1Min --initial_capital 10000 --start_date 2025-06-01 --end_date 2025-06-26 --enhanced --skip_optimization --compounded



===LIVE==>>>>>>  LIVE  SuperTRend BEST 062425  <<<<<<<------LIVE------->>>>>>>  
python supertrend_1d_live_trader.py

RUNNING: 06/26/25
python supertrend_1d_live_trader.py --symbol SOXL --timeframe 10Min --initial_capital 100000 --interval 30 --risk-per-trade 0.01 --interval 60

python supertrend_1d_live_trader.py --symbol TQQQ --timeframe 10Min --initial_capital 100000 --interval 30 --risk-per-trade 0.01 --interval 60

python supertrend_1d_live_trader.py --symbol CRCL --timeframe 10Min --initial_capital 100000 --interval 30 --risk-per-trade 0.01 --interval 60


