#!/usr/bin/env python3
"""
Enhanced SuperTrend + ML Trading Strategy
Combines EDA insights with SuperTrend signals for improved performance.

Key Features:
1. SuperTrend indicator signals
2. ML predictions from EDA analysis
3. Time-based choppiness filters
4. Risk management and position sizing
5. Performance tracking
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class EnhancedSuperTrendMLStrategy:
    """Enhanced SuperTrend strategy with ML integration"""
    
    def __init__(self, initial_capital=100000, risk_per_trade=0.02, 
                 ml_confidence_threshold=0.6, supertrend_weight=0.7, ml_weight=0.3):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.ml_confidence_threshold = ml_confidence_threshold
        self.supertrend_weight = supertrend_weight
        self.ml_weight = ml_weight
        self.position = None
        self.trades = []
        self.ml_model = None
        
    def calculate_supertrend(self, df, period=10, multiplier=3):
        """Calculate SuperTrend indicator"""
        # Calculate True Range (TR)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        # Calculate Basic Upper and Lower Bands
        df['basic_upper'] = (df['high'] + df['low']) / 2 + (multiplier * df['atr'])
        df['basic_lower'] = (df['high'] + df['low']) / 2 - (multiplier * df['atr'])
        
        # Calculate Final Upper and Lower Bands
        df['final_upper'] = df['basic_upper']
        df['final_lower'] = df['basic_lower']
        
        # Adjust bands based on previous values
        for i in range(1, len(df)):
            if df['close'].iloc[i] <= df['final_upper'].iloc[i-1]:
                df['final_upper'].iloc[i] = min(df['basic_upper'].iloc[i], df['final_upper'].iloc[i-1])
            else:
                df['final_upper'].iloc[i] = df['basic_upper'].iloc[i]
                
            if df['close'].iloc[i] >= df['final_lower'].iloc[i-1]:
                df['final_lower'].iloc[i] = max(df['basic_lower'].iloc[i], df['final_lower'].iloc[i-1])
            else:
                df['final_lower'].iloc[i] = df['basic_lower'].iloc[i]
        
        # Calculate SuperTrend
        df['supertrend'] = df['final_upper']
        df['supertrend_direction'] = 1  # 1 for uptrend, -1 for downtrend
        
        for i in range(1, len(df)):
            if df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and df['close'].iloc[i] <= df['final_upper'].iloc[i]:
                df['supertrend'].iloc[i] = df['final_upper'].iloc[i]
            elif df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and df['close'].iloc[i] > df['final_upper'].iloc[i]:
                df['supertrend'].iloc[i] = df['final_lower'].iloc[i]
                df['supertrend_direction'].iloc[i] = -1
            elif df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and df['close'].iloc[i] >= df['final_lower'].iloc[i]:
                df['supertrend'].iloc[i] = df['final_lower'].iloc[i]
            elif df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and df['close'].iloc[i] < df['final_lower'].iloc[i]:
                df['supertrend'].iloc[i] = df['final_upper'].iloc[i]
                df['supertrend_direction'].iloc[i] = 1
        
        # Clean up temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3', 'tr', 'basic_upper', 'basic_lower'], axis=1)
        
        return df
    
    def engineer_features(self, df):
        """Engineer features for ML model based on EDA insights"""
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['close'] / df['open']
        
        # Volume-based features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        
        # Volatility features
        df['volatility'] = df['price_change'].rolling(window=10).std()
        df['atr'] = df['high_low_ratio'].rolling(window=10).mean()
        
        # Moving averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        
        # Price position relative to moving averages
        df['price_vs_ma5'] = df['close'] / df['ma_5'] - 1
        df['price_vs_ma10'] = df['close'] / df['ma_10'] - 1
        df['price_vs_ma20'] = df['close'] / df['ma_20'] - 1
        
        # Time-based features (from EDA insights)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        
        # SuperTrend features
        df['price_vs_supertrend'] = df['close'] - df['supertrend']
        df['supertrend_distance'] = abs(df['price_vs_supertrend']) / df['close']
        
        # Target variable for classification (next bar direction)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        return df
    
    def train_ml_model(self, df):
        """Train machine learning model"""
        print("🤖 Training ML model...")
        
        # Select features for the model
        feature_columns = [
            'price_change', 'price_change_abs', 'high_low_ratio', 'open_close_ratio',
            'volume_change', 'volume_ratio', 'volatility', 'atr',
            'price_vs_ma5', 'price_vs_ma10', 'price_vs_ma20',
            'hour', 'day_of_week', 'is_market_open',
            'price_vs_supertrend', 'supertrend_distance'
        ]
        
        # Remove rows with missing values
        ml_data = df[feature_columns + ['target']].dropna()
        
        X = ml_data[feature_columns]
        y = ml_data['target']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ ML Model trained successfully")
        print(f"📊 Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📊 Baseline Accuracy: {max(y_test.mean(), 1-y_test.mean()):.4f} ({max(y_test.mean(), 1-y_test.mean())*100:.2f}%)")
        
        return model
    
    def get_combined_signal(self, supertrend_signal, ml_prediction, ml_confidence):
        """Combine SuperTrend and ML signals with weights"""
        # Convert ML prediction to -1/1
        ml_signal = 1 if ml_prediction == 1 else -1
        
        # Weighted combination
        combined_signal = (self.supertrend_weight * supertrend_signal + 
                          self.ml_weight * ml_signal * ml_confidence)
        
        # Determine final signal
        if combined_signal > 0.3:  # Threshold for long
            return 1, combined_signal
        elif combined_signal < -0.3:  # Threshold for short
            return -1, combined_signal
        else:
            return 0, combined_signal  # Neutral
    
    def calculate_position_size(self, entry_price, stop_loss_price):
        """Calculate position size based on risk management"""
        risk_per_share = abs(entry_price - stop_loss_price)
        risk_amount = self.capital * self.risk_per_trade
        shares = int(risk_amount / risk_per_share)
        return max(1, shares)  # Minimum 1 share
    
    def is_choppy_time(self, timestamp):
        """Check if current time is prone to choppiness (from EDA insights)"""
        hour = timestamp.hour
        day = timestamp.strftime('%A')
        
        # Avoid worst hours (from EDA analysis)
        avoid_hours = [9, 15, 21]  # Hours with highest direction change rates
        avoid_days = ['Saturday', 'Friday', 'Tuesday']
        
        if hour in avoid_hours:
            return True, f"Hour {hour} is in avoid list"
        
        if day in avoid_days:
            return True, f"{day} is in avoid list"
            
        return False, "Time OK"
    
    def run_strategy(self, df):
        """Run the enhanced SuperTrend + ML strategy"""
        print("🚀 Running Enhanced SuperTrend + ML Strategy...")
        
        # Calculate SuperTrend
        df = self.calculate_supertrend(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Train ML model
        self.ml_model = self.train_ml_model(df)
        
        # Feature columns for ML predictions
        feature_columns = [
            'price_change', 'price_change_abs', 'high_low_ratio', 'open_close_ratio',
            'volume_change', 'volume_ratio', 'volatility', 'atr',
            'price_vs_ma5', 'price_vs_ma10', 'price_vs_ma20',
            'hour', 'day_of_week', 'is_market_open',
            'price_vs_supertrend', 'supertrend_distance'
        ]
        
        # Strategy execution
        for i in range(20, len(df)):  # Start after enough data for indicators
            current_row = df.iloc[i]
            
            # Get SuperTrend signal
            supertrend_signal = current_row['supertrend_direction']
            
            # Get ML prediction
            features = current_row[feature_columns].values.reshape(1, -1)
            ml_prediction = self.ml_model.predict(features)[0]
            ml_confidence = max(self.ml_model.predict_proba(features)[0])
            
            # Get combined signal
            combined_signal, signal_strength = self.get_combined_signal(supertrend_signal, ml_prediction, ml_confidence)
            
            # Check choppiness filter
            skip_trade, filter_reason = self.is_choppy_time(current_row['timestamp'])
            
            # Trading logic
            if self.position is None:  # No position
                if not skip_trade and combined_signal != 0 and ml_confidence >= self.ml_confidence_threshold:
                    if combined_signal == 1:  # Long signal
                        entry_price = current_row['close']
                        stop_loss = entry_price * 0.95  # 5% stop loss
                        shares = self.calculate_position_size(entry_price, stop_loss)
                        
                        self.position = {
                            'side': 'long',
                            'entry_price': entry_price,
                            'entry_time': current_row['timestamp'],
                            'entry_idx': i,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'ml_confidence': ml_confidence,
                            'supertrend_signal': supertrend_signal,
                            'combined_signal': signal_strength
                        }
                        print(f"📈 LONG Entry: {current_row['timestamp']} @ ${entry_price:.2f}, {shares} shares")
                        print(f"   ML Confidence: {ml_confidence:.3f}, SuperTrend: {supertrend_signal}, Combined: {signal_strength:.3f}")
                        
                    elif combined_signal == -1:  # Short signal
                        entry_price = current_row['close']
                        stop_loss = entry_price * 1.05  # 5% stop loss
                        shares = self.calculate_position_size(entry_price, stop_loss)
                        
                        self.position = {
                            'side': 'short',
                            'entry_price': entry_price,
                            'entry_time': current_row['timestamp'],
                            'entry_idx': i,
                            'shares': shares,
                            'stop_loss': stop_loss,
                            'ml_confidence': ml_confidence,
                            'supertrend_signal': supertrend_signal,
                            'combined_signal': signal_strength
                        }
                        print(f"📉 SHORT Entry: {current_row['timestamp']} @ ${entry_price:.2f}, {shares} shares")
                        print(f"   ML Confidence: {ml_confidence:.3f}, SuperTrend: {supertrend_signal}, Combined: {signal_strength:.3f}")
                elif skip_trade:
                    print(f"⏭️ Skipped trade at {current_row['timestamp']}: {filter_reason}")
            
            else:  # Have position - check exit conditions
                current_price = current_row['close']
                bars_held = i - self.position['entry_idx']
                
                # Check stop loss
                if self.position['side'] == 'long' and current_price <= self.position['stop_loss']:
                    pnl = (current_price - self.position['entry_price']) * self.position['shares']
                    self.capital += pnl
                    self.trades.append({
                        'side': 'long',
                        'entry_time': self.position['entry_time'],
                        'entry_price': self.position['entry_price'],
                        'exit_time': current_row['timestamp'],
                        'exit_price': current_price,
                        'shares': self.position['shares'],
                        'pnl': pnl,
                        'exit_reason': 'stop_loss',
                        'holding_bars': bars_held,
                        'ml_confidence': self.position['ml_confidence'],
                        'supertrend_signal': self.position['supertrend_signal'],
                        'combined_signal': self.position['combined_signal']
                    })
                    print(f"🛑 LONG Stop Loss: {current_row['timestamp']} @ ${current_price:.2f}, PnL: ${pnl:.2f}")
                    self.position = None
                    
                elif self.position['side'] == 'short' and current_price >= self.position['stop_loss']:
                    pnl = (self.position['entry_price'] - current_price) * self.position['shares']
                    self.capital += pnl
                    self.trades.append({
                        'side': 'short',
                        'entry_time': self.position['entry_time'],
                        'entry_price': self.position['entry_price'],
                        'exit_time': current_row['timestamp'],
                        'exit_price': current_price,
                        'shares': self.position['shares'],
                        'pnl': pnl,
                        'exit_reason': 'stop_loss',
                        'holding_bars': bars_held,
                        'ml_confidence': self.position['ml_confidence'],
                        'supertrend_signal': self.position['supertrend_signal'],
                        'combined_signal': self.position['combined_signal']
                    })
                    print(f"🛑 SHORT Stop Loss: {current_row['timestamp']} @ ${current_price:.2f}, PnL: ${pnl:.2f}")
                    self.position = None
                
                # Check SuperTrend exit (after minimum holding period)
                elif bars_held >= 20:  # Minimum 20 bars holding
                    if (self.position['side'] == 'long' and supertrend_signal == -1) or \
                       (self.position['side'] == 'short' and supertrend_signal == 1):
                        
                        if self.position['side'] == 'long':
                            pnl = (current_price - self.position['entry_price']) * self.position['shares']
                        else:
                            pnl = (self.position['entry_price'] - current_price) * self.position['shares']
                        
                        self.capital += pnl
                        self.trades.append({
                            'side': self.position['side'],
                            'entry_time': self.position['entry_time'],
                            'entry_price': self.position['entry_price'],
                            'exit_time': current_row['timestamp'],
                            'exit_price': current_price,
                            'shares': self.position['shares'],
                            'pnl': pnl,
                            'exit_reason': 'supertrend_exit',
                            'holding_bars': bars_held,
                            'ml_confidence': self.position['ml_confidence'],
                            'supertrend_signal': self.position['supertrend_signal'],
                            'combined_signal': self.position['combined_signal']
                        })
                        print(f"🔄 {self.position['side'].upper()} SuperTrend Exit: {current_row['timestamp']} @ ${current_price:.2f}, PnL: ${pnl:.2f}")
                        self.position = None
        
        # Close any remaining position
        if self.position is not None:
            current_price = df.iloc[-1]['close']
            bars_held = len(df) - 1 - self.position['entry_idx']
            
            if self.position['side'] == 'long':
                pnl = (current_price - self.position['entry_price']) * self.position['shares']
            else:
                pnl = (self.position['entry_price'] - current_price) * self.position['shares']
            
            self.capital += pnl
            self.trades.append({
                'side': self.position['side'],
                'entry_time': self.position['entry_time'],
                'entry_price': self.position['entry_price'],
                'exit_time': df.iloc[-1]['timestamp'],
                'exit_price': current_price,
                'shares': self.position['shares'],
                'pnl': pnl,
                'exit_reason': 'end_of_data',
                'holding_bars': bars_held,
                'ml_confidence': self.position['ml_confidence'],
                'supertrend_signal': self.position['supertrend_signal'],
                'combined_signal': self.position['combined_signal']
            })
        
        # Convert trades to DataFrame
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            return trades_df, self.capital
        else:
            return pd.DataFrame(), self.capital

def main():
    """Main function to run the enhanced strategy"""
    print("🚀 Enhanced SuperTrend + ML Trading Strategy")
    print("=" * 50)
    
    # Load data
    import os
    cache_file = 'data/cache_SOXL_10Min.csv'
    if not os.path.exists(cache_file):
        print(f"❌ Cache file not found: {cache_file}")
        return
    
    # Load and prepare data
    df = pd.read_csv(cache_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} rows from {cache_file}")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Initialize strategy
    strategy = EnhancedSuperTrendMLStrategy(
        initial_capital=100000,
        risk_per_trade=0.02,
        ml_confidence_threshold=0.6,
        supertrend_weight=0.7,
        ml_weight=0.3
    )
    
    # Run strategy
    trades_df, final_capital = strategy.run_strategy(df)
    
    # Display results
    print(f"\n" + "="*60)
    print(f"📊 ENHANCED SUPERTREND + ML STRATEGY RESULTS")
    print(f"="*60)
    
    if not trades_df.empty:
        print(f"💰 Initial Capital: ${strategy.initial_capital:,.2f}")
        print(f"💰 Final Capital: ${final_capital:,.2f}")
        print(f"📈 Total PnL: ${final_capital - strategy.initial_capital:,.2f}")
        print(f"📊 Total Return: {((final_capital / strategy.initial_capital) - 1) * 100:.2f}%")
        print(f"🔄 Number of Trades: {len(trades_df)}")
        print(f"🎯 Win Rate: {(len(trades_df[trades_df['pnl'] > 0]) / len(trades_df) * 100):.1f}%")
        print(f"📊 Average Trade: ${trades_df['pnl'].mean():,.2f}")
        print(f"🤖 Average ML Confidence: {trades_df['ml_confidence'].mean():.3f}")
        print(f"⏱️ Average Holding Period: {trades_df['holding_bars'].mean():.1f} bars")
        
        # Exit reason analysis
        stop_loss_trades = len(trades_df[trades_df['exit_reason'] == 'stop_loss'])
        supertrend_exits = len(trades_df[trades_df['exit_reason'] == 'supertrend_exit'])
        print(f"🛑 Stop Loss Trades: {stop_loss_trades}")
        print(f"🔄 SuperTrend Exits: {supertrend_exits}")
        
        # Performance visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cumulative PnL
        cumulative_pnl = trades_df['pnl'].cumsum()
        ax1.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='blue')
        ax1.set_title('Cumulative PnL Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Trade Number', fontsize=12)
        ax1.set_ylabel('Cumulative PnL ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # PnL distribution
        ax2.hist(trades_df['pnl'], bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('PnL Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('PnL ($)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # ML Confidence vs PnL
        ax3.scatter(trades_df['ml_confidence'], trades_df['pnl'], alpha=0.6, color='purple')
        ax3.set_title('ML Confidence vs PnL', fontsize=14, fontweight='bold')
        ax3.set_xlabel('ML Confidence', fontsize=12)
        ax3.set_ylabel('PnL ($)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Holding period vs PnL
        ax4.scatter(trades_df['holding_bars'], trades_df['pnl'], alpha=0.6, color='orange')
        ax4.set_title('Holding Period vs PnL', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Holding Period (bars)', fontsize=12)
        ax4.set_ylabel('PnL ($)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    else:
        print("❌ No trades executed")

if __name__ == "__main__":
    main() 