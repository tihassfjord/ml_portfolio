"""
Stock Price Predictor using LSTM Neural Networks
Author: tihassfjord
GitHub: github.com/tihassfjord

This project implements a time series forecasting model for stock prices using LSTM
neural networks with technical indicators and multiple timeframe analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries, fallback to basic implementation if not available
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    ADVANCED_MODE = True
except ImportError:
    ADVANCED_MODE = False
    print("TensorFlow not available. Using simplified linear model.")

class StockPricePredictor:
    """
    A comprehensive stock price prediction system using LSTM neural networks
    with technical indicators and feature engineering.
    """
    
    def __init__(self, lookback_days=60, prediction_days=7):
        """
        Initialize the Stock Price Predictor
        
        Args:
            lookback_days (int): Number of historical days to use for prediction
            prediction_days (int): Number of days to predict into the future
        """
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler()
        self.model = None
        self.feature_columns = []
        
    def generate_synthetic_stock_data(self, days=1000, symbol="TECH"):
        """
        Generate realistic synthetic stock data with technical patterns
        
        Args:
            days (int): Number of days to generate
            symbol (str): Stock symbol name
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        np.random.seed(42)
        
        # Base price parameters
        initial_price = 100
        trend = 0.0003  # Slight upward trend
        volatility = 0.02
        
        # Generate price movements with realistic patterns
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        
        # Generate realistic price movements
        returns = np.random.normal(trend, volatility, days)
        
        # Add some market events (crashes and rallies)
        crash_points = np.random.choice(days, size=3, replace=False)
        rally_points = np.random.choice(days, size=5, replace=False)
        
        for point in crash_points:
            returns[point:point+5] = np.random.normal(-0.05, 0.01, 5)
        
        for point in rally_points:
            returns[point:point+3] = np.random.normal(0.03, 0.01, 3)
        
        # Calculate cumulative prices
        price_multipliers = np.cumprod(1 + returns)
        close_prices = initial_price * price_multipliers
        
        # Generate OHLC data
        high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, days)))
        low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, days)))
        
        # Ensure OHLC consistency
        open_prices = np.zeros(days)
        open_prices[0] = initial_price
        for i in range(1, days):
            open_prices[i] = close_prices[i-1] + np.random.normal(0, 0.005) * close_prices[i-1]
        
        # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
        for i in range(days):
            high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
            low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])
        
        # Generate volume (higher volume on significant price movements)
        base_volume = 1000000
        volume_multiplier = 1 + np.abs(returns) * 5
        volumes = np.random.normal(base_volume, base_volume * 0.3, days) * volume_multiplier
        volumes = np.maximum(volumes, 100000)  # Minimum volume
        
        # Create DataFrame
        stock_data = pd.DataFrame({
            'Date': dates,
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': close_prices,
            'Volume': volumes.astype(int),
            'Symbol': symbol
        })
        
        return stock_data
    
    def calculate_technical_indicators(self, data):
        """
        Calculate technical indicators for stock analysis
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with technical indicators added
        """
        df = data.copy()
        
        # Moving averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_21'] = df['Close'].rolling(window=21).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=21).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price change indicators
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_7d'] = df['Close'].pct_change(periods=7)
        df['High_Low_Ratio'] = df['High'] / df['Low']
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=21).std()
        
        return df
    
    def prepare_features(self, data):
        """
        Prepare features for machine learning model
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            
        Returns:
            tuple: (features, target, feature_names)
        """
        # Select features for modeling
        feature_cols = [
            'Close', 'Volume', 'MA_7', 'MA_21', 'MA_50',
            'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'RSI', 'BB_Position', 'BB_Width', 'Volume_Ratio',
            'Price_Change', 'Price_Change_7d', 'High_Low_Ratio', 'Volatility'
        ]
        
        # Remove rows with NaN values
        clean_data = data[feature_cols].dropna()
        
        # Prepare target (next day's closing price)
        target = clean_data['Close'].shift(-1).dropna()
        features = clean_data[:-1]  # Remove last row since target is shifted
        
        self.feature_columns = feature_cols
        
        return features.values, target.values, feature_cols
    
    def create_sequences(self, features, target):
        """
        Create sequences for LSTM training
        
        Args:
            features (np.array): Feature data
            target (np.array): Target data
            
        Returns:
            tuple: (X_sequences, y_sequences)
        """
        X, y = [], []
        
        for i in range(self.lookback_days, len(features)):
            X.append(features[i-self.lookback_days:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """
        Build LSTM neural network model
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            tensorflow.keras.Model: Compiled LSTM model
        """
        if not ADVANCED_MODE:
            return None
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def simple_linear_model(self, X_train, y_train):
        """
        Simple linear regression fallback when TensorFlow is not available
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            
        Returns:
            dict: Model coefficients
        """
        # Use simple linear regression on last values
        X_simple = X_train[:, -1, 0].reshape(-1, 1)  # Use last close price
        
        # Calculate linear regression coefficients
        X_mean = np.mean(X_simple)
        y_mean = np.mean(y_train)
        
        numerator = np.sum((X_simple.flatten() - X_mean) * (y_train - y_mean))
        denominator = np.sum((X_simple.flatten() - X_mean) ** 2)
        
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * X_mean
        
        return {'slope': slope, 'intercept': intercept}
    
    def train_model(self, data, train_split=0.8):
        """
        Train the stock price prediction model
        
        Args:
            data (pd.DataFrame): Stock data
            train_split (float): Proportion of data for training
            
        Returns:
            dict: Training history and metrics
        """
        print("Preparing data and calculating technical indicators...")
        
        # Calculate technical indicators
        data_with_indicators = self.calculate_technical_indicators(data)
        
        # Prepare features
        features, target, feature_names = self.prepare_features(data_with_indicators)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        split_index = int(len(features_scaled) * train_split)
        
        if ADVANCED_MODE:
            # Create sequences for LSTM
            X, y = self.create_sequences(features_scaled, target)
            
            X_train = X[:split_index - self.lookback_days]
            X_test = X[split_index - self.lookback_days:]
            y_train = y[:split_index - self.lookback_days]
            y_test = y[split_index - self.lookback_days:]
            
            print(f"Training LSTM model with {len(X_train)} samples...")
            
            # Build and train LSTM model
            self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            # Make predictions
            train_pred = self.model.predict(X_train)
            test_pred = self.model.predict(X_test)
            
        else:
            # Use simple linear model
            print("Training simple linear model...")
            
            X, y = self.create_sequences(features_scaled, target)
            
            X_train = X[:split_index - self.lookback_days]
            X_test = X[split_index - self.lookback_days:]
            y_train = y[:split_index - self.lookback_days]
            y_test = y[split_index - self.lookback_days:]
            
            self.model = self.simple_linear_model(X_train, y_train)
            
            # Make predictions
            train_pred = (X_train[:, -1, 0] * self.model['slope'] + self.model['intercept'])
            test_pred = (X_test[:, -1, 0] * self.model['slope'] + self.model['intercept'])
            
            history = {'loss': [0.1], 'val_loss': [0.1]}
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        metrics = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_pred': train_pred,
            'test_pred': test_pred,
            'y_train': y_train,
            'y_test': y_test
        }
        
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Testing MSE: {test_mse:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        
        return history, metrics
    
    def predict_future_prices(self, data, days=7):
        """
        Predict future stock prices
        
        Args:
            data (pd.DataFrame): Historical stock data
            days (int): Number of days to predict
            
        Returns:
            np.array: Predicted prices
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare recent data
        data_with_indicators = self.calculate_technical_indicators(data)
        features, _, _ = self.prepare_features(data_with_indicators)
        features_scaled = self.scaler.transform(features)
        
        # Get last sequence
        last_sequence = features_scaled[-self.lookback_days:]
        predictions = []
        
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            if ADVANCED_MODE:
                # LSTM prediction
                pred = self.model.predict(current_sequence.reshape(1, self.lookback_days, -1), verbose=0)
                next_price = pred[0, 0]
            else:
                # Simple linear model prediction
                last_close = current_sequence[-1, 0]  # Close price is first feature
                next_price = last_close * self.model['slope'] + self.model['intercept']
            
            predictions.append(next_price)
            
            # Update sequence (simplified - just update close price)
            new_row = current_sequence[-1].copy()
            new_row[0] = next_price  # Update close price
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return np.array(predictions)
    
    def plot_results(self, data, history, metrics):
        """
        Create comprehensive visualization of results
        
        Args:
            data (pd.DataFrame): Original stock data
            history: Training history
            metrics (dict): Training metrics
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Stock Price Prediction Analysis - tihassfjord', fontsize=16)
        
        # 1. Training History
        axes[0, 0].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Training History')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Predictions vs Actual
        axes[0, 1].scatter(metrics['y_test'], metrics['test_pred'], alpha=0.6)
        axes[0, 1].plot([metrics['y_test'].min(), metrics['y_test'].max()], 
                       [metrics['y_test'].min(), metrics['y_test'].max()], 'r--')
        axes[0, 1].set_title(f'Predictions vs Actual (R² = {metrics["test_r2"]:.3f})')
        axes[0, 1].set_xlabel('Actual Price')
        axes[0, 1].set_ylabel('Predicted Price')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Price Timeline with Predictions
        recent_data = data.tail(200).copy()
        axes[0, 2].plot(recent_data.index, recent_data['Close'], label='Actual Price', linewidth=2)
        
        # Add future predictions
        if len(metrics['test_pred']) > 0:
            test_start_idx = len(data) - len(metrics['test_pred'])
            pred_indices = range(test_start_idx, len(data))
            axes[0, 2].plot(pred_indices, metrics['test_pred'], 
                           label='Predictions', alpha=0.8, linestyle='--')
        
        axes[0, 2].set_title('Stock Price Timeline with Predictions')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Price')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Technical Indicators
        data_with_indicators = self.calculate_technical_indicators(data)
        recent_indicators = data_with_indicators.tail(100)
        
        axes[1, 0].plot(recent_indicators.index, recent_indicators['Close'], label='Close')
        axes[1, 0].plot(recent_indicators.index, recent_indicators['MA_21'], label='MA21')
        axes[1, 0].plot(recent_indicators.index, recent_indicators['BB_Upper'], 
                       label='BB Upper', alpha=0.7)
        axes[1, 0].plot(recent_indicators.index, recent_indicators['BB_Lower'], 
                       label='BB Lower', alpha=0.7)
        axes[1, 0].set_title('Technical Indicators')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. RSI
        axes[1, 1].plot(recent_indicators.index, recent_indicators['RSI'])
        axes[1, 1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        axes[1, 1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        axes[1, 1].set_title('RSI Indicator')
        axes[1, 1].set_ylabel('RSI')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Volume Analysis
        axes[1, 2].bar(recent_indicators.index, recent_indicators['Volume'], alpha=0.6)
        axes[1, 2].plot(recent_indicators.index, recent_indicators['Volume_MA'], 
                       color='red', label='Volume MA')
        axes[1, 2].set_title('Volume Analysis')
        axes[1, 2].set_ylabel('Volume')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Performance metrics summary
        print("\n" + "="*50)
        print("STOCK PRICE PREDICTION PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Model Type: {'LSTM Neural Network' if ADVANCED_MODE else 'Linear Regression'}")
        print(f"Training R²: {metrics['train_r2']:.4f}")
        print(f"Testing R²: {metrics['test_r2']:.4f}")
        print(f"Training MSE: {metrics['train_mse']:.4f}")
        print(f"Testing MSE: {metrics['test_mse']:.4f}")
        print(f"Training MAE: {np.mean(np.abs(metrics['y_train'] - metrics['train_pred'])):.4f}")
        print(f"Testing MAE: {np.mean(np.abs(metrics['y_test'] - metrics['test_pred'])):.4f}")
        
        # Future predictions
        try:
            future_predictions = self.predict_future_prices(data, days=7)
            print(f"\nFuture Price Predictions (Next 7 Days):")
            for i, price in enumerate(future_predictions, 1):
                print(f"Day {i}: ${price:.2f}")
        except Exception as e:
            print(f"Could not generate future predictions: {e}")

def main():
    """
    Main function to demonstrate stock price prediction
    """
    print("Stock Price Predictor with LSTM Neural Networks")
    print("Author: tihassfjord")
    print("=" * 60)
    
    # Initialize predictor
    predictor = StockPricePredictor(lookback_days=60, prediction_days=7)
    
    # Generate synthetic stock data
    print("Generating synthetic stock market data...")
    stock_data = predictor.generate_synthetic_stock_data(days=1000, symbol="TECH")
    
    print(f"Generated {len(stock_data)} days of stock data")
    print(f"Price range: ${stock_data['Close'].min():.2f} - ${stock_data['Close'].max():.2f}")
    
    # Train model
    print("\nTraining stock price prediction model...")
    history, metrics = predictor.train_model(stock_data, train_split=0.8)
    
    # Visualize results
    predictor.plot_results(stock_data, history, metrics)
    
    # Interactive prediction
    print("\n" + "="*60)
    print("INTERACTIVE STOCK ANALYSIS")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Generate new stock data and retrain")
        print("2. Predict future prices")
        print("3. Show technical analysis")
        print("4. Exit")
        
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                # Generate new data
                days = int(input("Enter number of days to generate (500-2000): ") or "1000")
                symbol = input("Enter stock symbol (default: TECH): ") or "TECH"
                
                new_data = predictor.generate_synthetic_stock_data(days=days, symbol=symbol)
                history, metrics = predictor.train_model(new_data, train_split=0.8)
                predictor.plot_results(new_data, history, metrics)
                
            elif choice == '2':
                # Predict future prices
                days = int(input("Enter number of days to predict (1-30): ") or "7")
                
                try:
                    predictions = predictor.predict_future_prices(stock_data, days=days)
                    print(f"\nPredicted prices for next {days} days:")
                    for i, price in enumerate(predictions, 1):
                        print(f"Day {i}: ${price:.2f}")
                        
                    # Simple visualization
                    plt.figure(figsize=(10, 6))
                    last_prices = stock_data['Close'].tail(30).values
                    plt.plot(range(-30, 0), last_prices, 'b-', label='Historical', linewidth=2)
                    plt.plot(range(0, days), predictions, 'r--', label='Predicted', linewidth=2)
                    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.7)
                    plt.title(f'Stock Price Prediction - {days} Days Ahead')
                    plt.xlabel('Days')
                    plt.ylabel('Price ($)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.show()
                    
                except Exception as e:
                    print(f"Error making predictions: {e}")
                
            elif choice == '3':
                # Show technical analysis
                data_with_indicators = predictor.calculate_technical_indicators(stock_data)
                latest = data_with_indicators.iloc[-1]
                
                print(f"\nLatest Technical Analysis:")
                print(f"Current Price: ${latest['Close']:.2f}")
                print(f"RSI: {latest['RSI']:.1f} ({'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral'})")
                print(f"MACD: {latest['MACD']:.3f}")
                print(f"Bollinger Band Position: {latest['BB_Position']:.2f}")
                print(f"Volume Ratio: {latest['Volume_Ratio']:.2f}")
                print(f"7-day Price Change: {latest['Price_Change_7d']:.1%}")
                
            elif choice == '4':
                print("Thank you for using Stock Price Predictor!")
                break
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
