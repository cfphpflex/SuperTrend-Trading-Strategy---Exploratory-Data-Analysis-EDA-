#!/usr/bin/env python3
"""
Enhanced SuperTrend Strategy with ML Integration
Baseline: $50K → $110K (120% return) - TARGET: IMPROVE THIS

ML Enhancements:
1. XGBoost Signal Validation - Use existing indicators as features
2. VIX-based Risk Management - Dynamic position sizing
3. ADX Trend Filtering - Only trade strong trends
4. Donchian Channel Confirmation - Validate breakouts

Goal: Increase final capital above $110,024.28 with reduced drawdown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.svm import SVC
import joblib

warnings.filterwarnings('ignore')

class EnhancedSuperTrendMLStrategy:
    """
    Enhanced SuperTrend strategy with ML integration
    Target: >$110,024.28 final capital with reduced drawdown
    """
    
    def __init__(self, 
                 initial_capital: float = 50000,
                 risk_per_trade: float = 0.01,  # 1% risk per trade
                 supertrend_period: int = 10,
                 supertrend_multiplier: float = 3.0):
        
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        
        # ML Models
        self.xgb_model = None
        self.vix_anomaly_model = None
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.trades = []
        self.signal_history = []
        self.current_position = None
        self.peak_capital = initial_capital
        
        print(f"🎯 Target: Improve on baseline $110,024.28 final capital")
        print(f"🤖 ML Enhancements: XGBoost + VIX + ADX + Donchian")
        
    def calculate_supertrend(self, df: pd.DataFrame, period: int = None, multiplier: float = None) -> pd.DataFrame:
        """Calculate SuperTrend indicator (baseline method) with optional period and multiplier for multi-timeframe support"""
        df = df.copy()
        
        # Use provided period/multiplier or defaults
        if period is None:
            period = self.supertrend_period
        if multiplier is None:
            multiplier = self.supertrend_multiplier
        
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
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index) for trend strength"""
        df = df.copy()
        
        # Calculate +DM and -DM
        df['high_diff'] = df['high'] - df['high'].shift(1)
        df['low_diff'] = df['low'].shift(1) - df['low']
        
        df['plus_dm'] = np.where((df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0), df['high_diff'], 0)
        df['minus_dm'] = np.where((df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0), df['low_diff'], 0)
        
        # Calculate smoothed values
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr'])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr'])
        
        # Calculate DX and ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        # Clean up temporary columns
        df = df.drop(['high_diff', 'low_diff', 'plus_dm', 'minus_dm', 'dx'], axis=1)
        
        return df
    
    def calculate_donchian_channels(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Donchian Channels for support/resistance"""
        df = df.copy()
        
        df['donchian_upper'] = df['high'].rolling(window=period).max()
        df['donchian_lower'] = df['low'].rolling(window=period).min()
        df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2
        
        # Calculate breakout signals
        df['upper_breakout'] = (df['close'] - df['donchian_upper']) / df['atr']
        df['lower_breakout'] = (df['donchian_lower'] - df['close']) / df['atr']
        
        return df
    
    def calculate_vix_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VIX-based features for risk management"""
        df = df.copy()
        
        # Simulate VIX (using price volatility as proxy)
        df['price_change'] = df['close'].pct_change()
        df['vix_proxy'] = df['price_change'].rolling(window=20).std() * np.sqrt(252) * 100
        
        # VIX-based features
        df['vix_ma'] = df['vix_proxy'].rolling(window=20).mean()
        df['vix_ratio'] = df['vix_proxy'] / df['vix_ma']
        df['vix_regime'] = pd.cut(df['vix_proxy'], 
                                 bins=[0, 15, 25, 50, 100], 
                                 labels=['low', 'normal', 'high', 'extreme'])
        
        return df
    
    def create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML features from technical indicators"""
        df = df.copy()
        
        # Multi-timeframe SuperTrend
        df['supertrend_5'] = self.calculate_supertrend(df, period=5)['supertrend_direction']
        df['supertrend_10'] = self.calculate_supertrend(df, period=10)['supertrend_direction']
        df['supertrend_20'] = self.calculate_supertrend(df, period=20)['supertrend_direction']
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_trend'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
        
        # Price momentum features
        df['price_momentum'] = df['close'].pct_change(5)
        df['price_acceleration'] = df['price_momentum'].diff()
        
        # ATR features
        df['atr_ratio'] = df['atr'] / df['atr'].rolling(window=20).mean()
        df['atr_trend'] = df['atr'].rolling(window=5).mean() / df['atr'].rolling(window=20).mean()
        
        # ADX trend strength
        adx_cat = pd.cut(df['adx'], 
                        bins=[0, 25, 50, 100], 
                        labels=[0, 1, 2], 
                        include_lowest=True)
        first_cat = adx_cat.cat.categories[0]
        df['adx_strength'] = adx_cat.fillna(first_cat).astype(int)
        
        # Donchian features
        df['donchian_position'] = (df['close'] - df['donchian_lower']) / (df['donchian_upper'] - df['donchian_lower'])
        df['donchian_width'] = (df['donchian_upper'] - df['donchian_lower']) / df['close']
        
        return df
    
    def train_xgboost_model(self, df: pd.DataFrame) -> None:
        """Train XGBoost model for signal validation"""
        print("🤖 Training XGBoost model for signal validation...")
        
        # Create features
        feature_columns = [
            'supertrend_5', 'supertrend_10', 'supertrend_20',
            'volume_ratio', 'volume_trend',
            'price_momentum', 'price_acceleration',
            'atr_ratio', 'atr_trend',
            'adx_strength', 'plus_di', 'minus_di',
            'donchian_position', 'donchian_width',
            'upper_breakout', 'lower_breakout',
            'vix_ratio'
        ]
        
        # Create target (future returns)
        df['future_return'] = df['close'].shift(-1) / df['close'] - 1
        df['target'] = np.where(df['future_return'] > 0.005, 1, 0)  # 0.5% threshold
        
        # Remove NaN values
        df_clean = df.dropna()
        
        if len(df_clean) < 100:
            print("⚠️ Insufficient data for ML training, using baseline signals")
            return
        
        # Prepare features and target
        X = df_clean[feature_columns].fillna(0)
        y = df_clean['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.xgb_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.xgb_model.score(X_train_scaled, y_train)
        test_score = self.xgb_model.score(X_test_scaled, y_test)
        
        print(f"✅ XGBoost trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
    
    def train_vix_anomaly_model(self, df: pd.DataFrame) -> None:
        """Train VIX anomaly detection model"""
        print("🛡️ Training VIX anomaly detection model...")
        
        # Use VIX proxy for anomaly detection
        vix_data = df['vix_proxy'].dropna().values.reshape(-1, 1)
        
        if len(vix_data) < 50:
            print("⚠️ Insufficient VIX data, using baseline risk management")
            return
        
        self.vix_anomaly_model = IsolationForest(
            contamination=0.1,  # 10% of data as anomalies
            random_state=42
        )
        
        self.vix_anomaly_model.fit(vix_data)
        print("✅ VIX anomaly model trained")
    
    def get_ml_enhanced_signal(self, df: pd.DataFrame, i: int) -> Tuple[int, float, float]:
        """Get ML-enhanced trading signal"""
        if i < 50:  # Need enough data for ML features
            return 0, 0.0, 1.0
        
        current_row = df.iloc[i]
        
        # Base SuperTrend signal
        supertrend_signal = current_row['supertrend_direction']
        
        # 1. ADX Trend Filtering - Only trade strong trends
        adx_strength = current_row.get('adx_strength', 0)
        if adx_strength < 1:  # Weak trend
            return 0, 0.0, 1.0
        
        # 2. Donchian Channel Confirmation
        donchian_position = current_row.get('donchian_position', 0.5)
        upper_breakout = current_row.get('upper_breakout', 0)
        lower_breakout = current_row.get('lower_breakout', 0)
        
        # 3. XGBoost Signal Validation
        ml_confidence = 0.5  # Default confidence
        
        if self.xgb_model is not None:
            try:
                # Prepare features for ML prediction
                feature_columns = [
                    'supertrend_5', 'supertrend_10', 'supertrend_20',
                    'volume_ratio', 'volume_trend',
                    'price_momentum', 'price_acceleration',
                    'atr_ratio', 'atr_trend',
                    'adx_strength', 'plus_di', 'minus_di',
                    'donchian_position', 'donchian_width',
                    'upper_breakout', 'lower_breakout',
                    'vix_ratio'
                ]
                
                features = current_row[feature_columns].fillna(0).values.reshape(1, -1)
                features_scaled = self.scaler.transform(features)
                
                # Get ML prediction and confidence
                ml_prediction = self.xgb_model.predict(features_scaled)[0]
                ml_confidence = self.xgb_model.predict_proba(features_scaled)[0].max()
                
            except Exception as e:
                print(f"⚠️ ML prediction error: {e}")
                ml_confidence = 0.5
        
        # 4. VIX-based Risk Management
        risk_multiplier = 1.0
        if self.vix_anomaly_model is not None:
            try:
                vix_value = current_row.get('vix_proxy', 20)
                vix_anomaly = self.vix_anomaly_model.predict([[vix_value]])[0]
                
                if vix_anomaly == -1:  # Anomaly detected
                    risk_multiplier = 0.5  # Reduce position size
                elif vix_value > 30:  # High volatility
                    risk_multiplier = 0.7
                elif vix_value < 15:  # Low volatility
                    risk_multiplier = 1.2
                    
            except Exception as e:
                print(f"⚠️ VIX risk calculation error: {e}")
        
        # Combine signals
        signal_strength = 0.0
        
        # SuperTrend base signal
        if supertrend_signal == 1:
            signal_strength += 0.3
        elif supertrend_signal == -1:
            signal_strength -= 0.3
        
        # Donchian confirmation
        if supertrend_signal == 1 and upper_breakout > 0.5:
            signal_strength += 0.2
        elif supertrend_signal == -1 and lower_breakout > 0.5:
            signal_strength -= 0.2
        
        # ML confidence boost
        if ml_confidence > 0.7:
            signal_strength *= 1.5
        elif ml_confidence < 0.3:
            signal_strength *= 0.5
        
        # Determine final signal
        if signal_strength > 0.3:
            final_signal = 1  # Long
        elif signal_strength < -0.3:
            final_signal = -1  # Short
        else:
            final_signal = 0  # Neutral
        
        return final_signal, signal_strength, risk_multiplier
    
    def calculate_dynamic_stop_loss(self, entry_price: float, direction: int, atr: float, vix_ratio: float) -> float:
        """Calculate dynamic stop loss based on ATR and VIX"""
        # Base ATR stop distance
        base_stop_distance = atr * 1.5
        
        # VIX-adjusted stop distance
        if vix_ratio > 1.5:  # High volatility
            stop_distance = base_stop_distance * 1.2
        elif vix_ratio < 0.8:  # Low volatility
            stop_distance = base_stop_distance * 0.8
        else:
            stop_distance = base_stop_distance
        
        if direction == 1:  # Long
            stop_loss = entry_price - stop_distance
        else:  # Short
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, risk_multiplier: float) -> int:
        """Calculate position size with ML-enhanced risk management"""
        # Base position size
        risk_per_share = abs(entry_price - stop_loss)
        base_risk_amount = self.capital * self.risk_per_trade
        base_shares = int(base_risk_amount / risk_per_share)
        
        # ML risk adjustment
        adjusted_shares = int(base_shares * risk_multiplier)
        
        # Additional drawdown protection
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if current_drawdown > 0.1:  # If drawdown > 10%
            adjusted_shares = int(adjusted_shares * 0.7)
        elif current_drawdown > 0.05:  # If drawdown > 5%
            adjusted_shares = int(adjusted_shares * 0.85)
        
        return max(1, adjusted_shares)
    
    def run_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run the enhanced ML strategy"""
        print("🚀 Running Enhanced SuperTrend ML Strategy...")
        
        # Calculate all technical indicators
        df = self.calculate_supertrend(df)
        df = self.calculate_adx(df)
        df = self.calculate_donchian_channels(df)
        df = self.calculate_vix_features(df)
        df = self.create_ml_features(df)
        
        # Train ML models
        self.train_xgboost_model(df)
        self.train_vix_anomaly_model(df)
        
        # Initialize strategy variables
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.trades = []
        self.signal_history = []
        self.current_position = None
        
        # Run strategy
        for i in range(50, len(df)):  # Start after ML features are available
            current_row = df.iloc[i]
            current_price = current_row['close']
            
            # Update peak capital
            if self.capital > self.peak_capital:
                self.peak_capital = self.capital
            
            # Get ML-enhanced signal
            signal, signal_strength, risk_multiplier = self.get_ml_enhanced_signal(df, i)
            
            # Execute trades
            if self.current_position is None:  # No position
                if signal != 0:
                    self.execute_ml_trade(signal, current_row, signal_strength, risk_multiplier)
            else:  # Have position
                # Check stop loss or take profit
                if self.check_exit_conditions(current_row):
                    self.close_position(current_price, self.current_position['type'])
                # Check for signal reversal
                elif signal != 0 and signal != self.current_position['type']:
                    self.close_position(current_price, self.current_position['type'])
                    self.execute_ml_trade(signal, current_row, signal_strength, risk_multiplier)
            
            # Record signal
            self.signal_history.append({
                'date': current_row.name,
                'price': current_price,
                'signal': signal,
                'signal_strength': signal_strength,
                'risk_multiplier': risk_multiplier,
                'capital': self.capital
            })
        
        # Close any remaining position
        if self.current_position is not None:
            self.close_position(df.iloc[-1]['close'], self.current_position['type'])
        
        # Calculate performance
        performance = self.calculate_performance_metrics()
        self.print_performance_summary(performance)
        
        return performance
    
    def execute_ml_trade(self, signal: int, current_row: pd.Series, signal_strength: float, risk_multiplier: float):
        """Execute trade with ML enhancements"""
        entry_price = current_row['close']
        atr = current_row['atr']
        vix_ratio = current_row.get('vix_ratio', 1.0)
        
        # Calculate dynamic stop loss
        stop_loss = self.calculate_dynamic_stop_loss(entry_price, signal, atr, vix_ratio)
        
        # Calculate position size with ML risk management
        shares = self.calculate_position_size(entry_price, stop_loss, risk_multiplier)
        
        # Calculate trade value
        trade_value = shares * entry_price
        
        if trade_value > self.capital:
            shares = int(self.capital / entry_price)
            trade_value = shares * entry_price
        
        # Record trade
        position_type = 'long' if signal == 1 else 'short'
        
        self.current_position = {
            'type': position_type,
            'entry_price': entry_price,
            'shares': shares,
            'stop_loss': stop_loss,
            'entry_date': current_row.name,
            'signal_strength': signal_strength,
            'risk_multiplier': risk_multiplier
        }
        
        # Update capital
        self.capital -= trade_value
        
        print(f"📈 {position_type.upper()} Entry: {shares} shares @ ${entry_price:.2f}")
        print(f"   Stop Loss: ${stop_loss:.2f}, Signal Strength: {signal_strength:.2f}")
        print(f"   Risk Multiplier: {risk_multiplier:.2f}, Capital: ${self.capital:.2f}")
    
    def check_exit_conditions(self, current_row: pd.Series) -> bool:
        """Check if position should be closed"""
        if self.current_position is None:
            return False
        
        current_price = current_row['close']
        stop_loss = self.current_position['stop_loss']
        position_type = self.current_position['type']
        
        # Stop loss hit
        if position_type == 'long' and current_price <= stop_loss:
            return True
        elif position_type == 'short' and current_price >= stop_loss:
            return True
        
        # Take profit (2:1 risk-reward)
        entry_price = self.current_position['entry_price']
        if position_type == 'long':
            take_profit = entry_price + (entry_price - stop_loss) * 2
            if current_price >= take_profit:
                return True
        else:  # short
            take_profit = entry_price - (stop_loss - entry_price) * 2
            if current_price <= take_profit:
                return True
        
        return False
    
    def close_position(self, current_price: float, position_type: str):
        """Close current position"""
        if self.current_position is None:
            return
        
        shares = self.current_position['shares']
        entry_price = self.current_position['entry_price']
        entry_date = self.current_position['entry_date']
        
        # Calculate P&L
        if position_type == 'long':
            pnl = (current_price - entry_price) * shares
        else:  # short
            pnl = (entry_price - current_price) * shares
        
        # Update capital
        trade_value = shares * current_price
        self.capital += trade_value + pnl
        
        # Record trade
        self.trades.append({
            'entry_date': entry_date,
            'exit_date': pd.Timestamp.now(),
            'type': position_type,
            'entry_price': entry_price,
            'exit_price': current_price,
            'shares': shares,
            'pnl': pnl,
            'return_pct': (pnl / (entry_price * shares)) * 100
        })
        
        print(f"📉 {position_type.upper()} Exit: {shares} shares @ ${current_price:.2f}")
        print(f"   P&L: ${pnl:.2f} ({self.trades[-1]['return_pct']:.2f}%)")
        print(f"   Capital: ${self.capital:.2f}")
        
        self.current_position = None
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {
                'final_capital': self.capital,
                'total_return': 0,
                'total_return_pct': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }
        
        # Basic metrics
        final_capital = self.capital
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100
        num_trades = len(self.trades)
        
        # Win/loss analysis
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Drawdown calculation
        capital_curve = [self.initial_capital]
        for trade in self.trades:
            capital_curve.append(capital_curve[-1] + trade['pnl'])
        
        max_drawdown = self.calculate_max_drawdown(capital_curve)
        
        # Risk metrics
        returns = [t['return_pct'] for t in self.trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Profit factor
        gross_profit = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
        gross_loss = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'final_capital': final_capital,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'trades': self.trades
        }
    
    def calculate_max_drawdown(self, capital_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = capital_curve[0]
        max_dd = 0
        
        for capital in capital_curve:
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd * 100
    
    def print_performance_summary(self, performance: Dict[str, Any]):
        """Print detailed performance summary"""
        print("\n" + "="*60)
        print("🎯 ENHANCED ML STRATEGY PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"💰 Final Capital: ${performance['final_capital']:,.2f}")
        print(f"📈 Total Return: ${performance['total_return']:,.2f} ({performance['total_return_pct']:.2f}%)")
        print(f"🎯 Baseline Target: $110,024.28")
        print(f"✅ Target Achieved: {'YES' if performance['final_capital'] > 110024.28 else 'NO'}")
        
        print(f"\n📊 Trading Statistics:")
        print(f"   Number of Trades: {performance['num_trades']}")
        print(f"   Win Rate: {performance['win_rate']:.1f}%")
        print(f"   Average Win: ${performance['avg_win']:.2f}")
        print(f"   Average Loss: ${performance['avg_loss']:.2f}")
        
        print(f"\n🛡️ Risk Metrics:")
        print(f"   Maximum Drawdown: {performance['max_drawdown']:.2f}%")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"   Profit Factor: {performance['profit_factor']:.2f}")
        
        # Compare with baseline
        baseline_return = 110024.28 - 50000
        current_return = performance['total_return']
        improvement = current_return - baseline_return
        
        print(f"\n📈 Performance vs Baseline:")
        print(f"   Baseline Return: ${baseline_return:,.2f}")
        print(f"   Current Return: ${current_return:,.2f}")
        print(f"   Improvement: ${improvement:,.2f} ({improvement/baseline_return*100:.1f}%)")
        
        print("="*60)

def main():
    """Main execution function"""
    print("🚀 Enhanced SuperTrend ML Strategy")
    print("="*50)
    
    # Load data
    try:
        df = pd.read_csv('data/cache_SOXL_10Min.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        print(f"✅ Data loaded: {len(df)} records from {df.index[0]} to {df.index[-1]}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run enhanced strategy
    strategy = EnhancedSuperTrendMLStrategy()
    performance = strategy.run_strategy(df)
    
    # Save results
    results_df = pd.DataFrame(strategy.trades)
    if not results_df.empty:
        results_df.to_csv('enhanced_ml_results.csv', index=False)
        print(f"✅ Results saved to enhanced_ml_results.csv")

if __name__ == "__main__":
    main() 