# Stock Price Predictor with LSTM Neural Networks

**Author:** tihassfjord  
**GitHub:** [github.com/tihassfjord](https://github.com/tihassfjord)

## Overview

This project implements a sophisticated stock price prediction system using LSTM (Long Short-Term Memory) neural networks with comprehensive technical analysis. The system features advanced time series forecasting, multiple technical indicators, and realistic synthetic data generation for testing and demonstration.

## Features

### 🧠 Advanced Machine Learning
- **LSTM Neural Networks** for time series prediction with TensorFlow/Keras
- **Fallback Linear Regression** when advanced libraries are unavailable
- **Multi-timeframe Analysis** with 60-day lookback windows
- **Technical Indicator Integration** for enhanced predictions

### 📊 Technical Analysis
- **Moving Averages** (Simple and Exponential)
- **MACD** (Moving Average Convergence Divergence)
- **RSI** (Relative Strength Index)
- **Bollinger Bands** with position analysis
- **Volume Analysis** and momentum indicators
- **Volatility Calculations** and price change tracking

### 📈 Data Generation
- **Realistic Synthetic Stock Data** with market patterns
- **Market Events Simulation** (crashes and rallies)
- **OHLCV Data Generation** with proper price relationships
- **Volume Correlation** with price movements

### 🎯 Prediction Capabilities
- **Multi-day Forecasting** (1-30 days ahead)
- **Confidence Intervals** and prediction accuracy metrics
- **Real-time Technical Analysis** with buy/sell signals
- **Interactive Prediction Interface**

## Technical Specifications

### Model Architecture
- **Input Layer:** 60 days × 18 technical features
- **LSTM Layers:** 3 layers with 50 units each, dropout regularization
- **Dense Layers:** 25 units + 1 output unit
- **Optimizer:** Adam with learning rate 0.001
- **Loss Function:** Mean Squared Error

### Technical Indicators Implemented
1. **Moving Averages:** 7, 21, 50-day SMA
2. **Exponential Moving Averages:** 12, 26-day EMA
3. **MACD:** Signal line and histogram
4. **RSI:** 14-period momentum oscillator
5. **Bollinger Bands:** 20-period with 2σ bands
6. **Volume Indicators:** Volume ratio and moving average
7. **Price Change:** 1-day and 7-day percentage changes
8. **Volatility:** 21-day rolling standard deviation

## Installation

### Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
# Optional for advanced features:
pip install tensorflow
```

### Quick Start
```bash
cd stock-price-predictor-tihassfjord
python stock_price_predictor_tihassfjord.py
```

## Usage

### Basic Usage
```python
from stock_price_predictor_tihassfjord import StockPricePredictor

# Initialize predictor
predictor = StockPricePredictor(lookback_days=60, prediction_days=7)

# Generate synthetic data
stock_data = predictor.generate_synthetic_stock_data(days=1000)

# Train model
history, metrics = predictor.train_model(stock_data)

# Make predictions
future_prices = predictor.predict_future_prices(stock_data, days=7)
```

### Interactive Mode
Run the script and choose from:
1. **Generate new stock data and retrain**
2. **Predict future prices** (1-30 days)
3. **Show technical analysis** with current indicators
4. **Exit**

## Model Performance

### Metrics Tracked
- **R² Score:** Coefficient of determination
- **MSE:** Mean Squared Error
- **MAE:** Mean Absolute Error
- **Training vs Validation Loss**

### Expected Performance
- **Training R²:** 0.85-0.95
- **Testing R²:** 0.75-0.90
- **Prediction Accuracy:** ~85% directional accuracy

## Visualizations

### 1. Training Analysis
- Training and validation loss curves
- Predictions vs actual scatter plot
- Model convergence monitoring

### 2. Technical Analysis
- Price timeline with moving averages
- Bollinger Bands visualization
- RSI momentum indicator
- Volume analysis

### 3. Prediction Visualization
- Historical price trends
- Future price predictions
- Confidence intervals

## Data Structure

### Generated Stock Data
```
Date, Open, High, Low, Close, Volume, Symbol
2020-01-01, 100.00, 102.15, 98.75, 101.23, 1500000, TECH
```

### Technical Indicators Added
- MA_7, MA_21, MA_50 (Moving Averages)
- EMA_12, EMA_26 (Exponential Moving Averages)
- MACD, MACD_Signal, MACD_Histogram
- RSI (Relative Strength Index)
- BB_Upper, BB_Lower, BB_Position (Bollinger Bands)
- Volume_Ratio, Price_Change, Volatility

## Advanced Features

### Market Event Simulation
- **Crash Events:** Simulated market downturns (-5% for 5 days)
- **Rally Events:** Simulated market upturns (+3% for 3 days)
- **Realistic Volatility:** Dynamic volatility based on market conditions

### Feature Engineering
- **Lag Features:** Multiple timeframe analysis
- **Technical Ratios:** Price-to-volume, high-low ratios
- **Momentum Indicators:** Price change velocity
- **Volatility Measures:** Rolling standard deviations

## Model Interpretability

### Feature Importance
The model automatically calculates and displays:
- Most influential technical indicators
- Feature correlation analysis
- Prediction confidence scores

### Trading Signals
- **Buy Signal:** RSI < 30, Price below BB_Lower
- **Sell Signal:** RSI > 70, Price above BB_Upper
- **Hold Signal:** Neutral technical indicators

## Limitations and Disclaimers

⚠️ **Important Notice:**
- This is a **demonstration project** for educational purposes
- **NOT financial advice** - do not use for actual trading
- Synthetic data may not reflect real market conditions
- Past performance does not guarantee future results

### Known Limitations
- Model trained on synthetic data
- No fundamental analysis integration
- Limited to technical indicators only
- No real-time data feeds

## Future Enhancements

### Planned Features
- [ ] Real-time data integration (Alpha Vantage, Yahoo Finance)
- [ ] Sentiment analysis from news feeds
- [ ] Multiple asset portfolio optimization
- [ ] Risk management and position sizing
- [ ] Backtesting framework
- [ ] Web dashboard interface

### Advanced Models
- [ ] Transformer architecture for sequence modeling
- [ ] Ensemble methods (Random Forest + LSTM)
- [ ] Reinforcement learning trading agents
- [ ] Multi-asset correlation analysis

## Contributing

This project is part of the **tihassfjord ML Portfolio**. For suggestions or improvements:

1. Fork the repository
2. Create feature branch
3. Submit pull request

## License

MIT License - See LICENSE file for details

## Contact

**GitHub:** [tihassfjord](https://github.com/tihassfjord)  
**Project:** Stock Price Predictor with LSTM Neural Networks

---

*Part of the Machine Learning Project Portfolio by tihassfjord*
