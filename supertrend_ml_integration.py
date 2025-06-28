#!/usr/bin/env python3
"""
Advanced SuperTrend + ML Trading Strategy Integration
Focused on Hyperparameter Tuning and Strategy Integration

Key Features:
1. Advanced hyperparameter tuning with Optuna
2. XGBoost and Neural Network optimization
3. SuperTrend + ML signal integration
4. Dynamic position sizing based on model confidence
5. Performance monitoring and backtesting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import joblib
import optuna
from typing import Dict, List, Tuple, Optional, Any

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

class AdvancedSuperTrendMLIntegration:
    """
    Advanced ML integration for SuperTrend trading strategy
    Focused on hyperparameter tuning and signal integration
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 risk_per_trade: float = 0.02,
                 ml_confidence_threshold: float = 0.65,
                 supertrend_weight: float = 0.6,
                 ml_weight: float = 0.4):
        
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.ml_confidence_threshold = ml_confidence_threshold
        self.supertrend_weight = supertrend_weight
        self.ml_weight = ml_weight
        
        # Model storage
        self.models = {}
        self.best_params = {}
        self.performance_metrics = {}
        
        # Performance tracking
        self.trades = []
        self.signal_history = []
        self.current_position = None
        
    def calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """
        Calculate SuperTrend indicator
        """
        df = df.copy()
        
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
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for ML models based on EDA insights
        """
        df = df.copy()
        
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
        
        # Time-based features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        
        # SuperTrend features
        if 'supertrend' in df.columns:
            df['price_vs_supertrend'] = df['close'] - df['supertrend']
            df['supertrend_distance'] = abs(df['price_vs_supertrend']) / df['close']
        
        # Target variable for classification (next bar direction)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        return df
    
    def optimize_xgboost_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters using Optuna
        """
        print("🔧 Optimizing XGBoost hyperparameters...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
            }
            
            model = XGBClassifier(**params, random_state=42, eval_metric='logloss', use_label_encoder=False)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        print(f"✅ Best XGBoost score: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_neural_network_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Optimize Neural Network hyperparameters using Optuna
        """
        print("🔧 Optimizing Neural Network hyperparameters...")
        
        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 1, 3)
            layers = []
            for i in range(n_layers):
                layers.append(trial.suggest_int(f'n_units_l{i}', 10, 200))
            
            params = {
                'hidden_layer_sizes': tuple(layers),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                'max_iter': 500
            }
            
            model = MLPClassifier(**params, random_state=42)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        
        print(f"✅ Best Neural Network score: {study.best_value:.4f}")
        return study.best_params
    
    def train_optimized_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train models with optimized hyperparameters
        """
        print("🤖 Training optimized ML models...")
        
        # Engineer features
        df_engineered = self.engineer_features(df)
        
        # Select features
        feature_columns = [
            'price_change', 'price_change_abs', 'high_low_ratio', 'open_close_ratio',
            'volume_change', 'volume_ratio', 'volatility', 'atr',
            'price_vs_ma5', 'price_vs_ma10', 'price_vs_ma20',
            'hour', 'day_of_week', 'is_market_open'
        ]
        
        # Add SuperTrend features if available
        if 'price_vs_supertrend' in df_engineered.columns:
            feature_columns.extend(['price_vs_supertrend', 'supertrend_distance'])
        
        # Remove rows with missing values
        ml_data = df_engineered[feature_columns + ['target']].dropna()
        X = ml_data[feature_columns]
        y = ml_data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Optimize and train XGBoost
        xgb_params = self.optimize_xgboost_hyperparameters(X_train, y_train)
        xgb_model = XGBClassifier(**xgb_params, random_state=42, eval_metric='logloss', use_label_encoder=False)
        xgb_model.fit(X_train, y_train)
        
        # Optimize and train Neural Network
        nn_params = self.optimize_neural_network_hyperparameters(X_train, y_train)
        nn_model = MLPClassifier(**nn_params, random_state=42)
        nn_model.fit(X_train, y_train)
        
        # Train Random Forest (baseline)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Evaluate models
        models = {
            'xgboost': xgb_model,
            'neural_network': nn_model,
            'random_forest': rf_model
        }
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            self.performance_metrics[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"✅ {name}: Accuracy={accuracy:.4f}, AUC={auc:.4f}")
        
        self.models = models
        self.best_params = {
            'xgboost': xgb_params,
            'neural_network': nn_params
        }
        
        return models
    
    def get_ensemble_prediction(self, features: pd.DataFrame) -> Tuple[int, float]:
        """
        Get ensemble prediction from all models
        """
        predictions = []
        probabilities = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(features)[0]
                proba = model.predict_proba(features)[0, 1]
                predictions.append(pred)
                probabilities.append(proba)
            except Exception as e:
                print(f"Warning: Error in {name} prediction: {e}")
                continue
        
        if not predictions:
            return 0, 0.5
        
        # Weighted ensemble based on AUC performance
        weights = []
        for name in self.models.keys():
            if name in self.performance_metrics:
                weights.append(self.performance_metrics[name]['auc'])
            else:
                weights.append(0.5)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1/len(predictions)] * len(predictions)
        
        # Calculate weighted probability
        weighted_proba = sum(p * w for p, w in zip(probabilities, weights))
        ensemble_prediction = 1 if weighted_proba > 0.5 else 0
        
        return ensemble_prediction, weighted_proba
    
    def integrate_supertrend_ml_signals(self, supertrend_signal: int, ml_prediction: int, 
                                      ml_confidence: float, market_conditions: Dict[str, Any]) -> Tuple[int, float, Dict[str, Any]]:
        """
        Integrate SuperTrend and ML signals with advanced logic
        """
        # Market condition adjustments
        volatility = market_conditions.get('volatility', 0.01)
        volume_ratio = market_conditions.get('volume_ratio', 1.0)
        
        # Dynamic weight adjustment based on market conditions
        supertrend_weight = self.supertrend_weight
        ml_weight = self.ml_weight
        
        # Adjust weights based on volatility
        if volatility > 0.02:  # High volatility - favor SuperTrend
            supertrend_weight *= 1.2
            ml_weight *= 0.8
        elif volatility < 0.005:  # Low volatility - favor ML
            supertrend_weight *= 0.8
            ml_weight *= 1.2
        
        # Adjust weights based on volume
        if volume_ratio < 0.8:  # Low volume - favor SuperTrend
            supertrend_weight *= 1.1
            ml_weight *= 0.9
        
        # Normalize weights
        total_weight = supertrend_weight + ml_weight
        supertrend_weight /= total_weight
        ml_weight /= total_weight
        
        # Convert signals to numeric
        supertrend_numeric = supertrend_signal
        ml_numeric = 1 if ml_prediction == 1 else -1
        
        # Weighted combination
        combined_signal = (supertrend_weight * supertrend_numeric + 
                          ml_weight * ml_numeric * ml_confidence)
        
        # Determine final signal with confidence threshold
        if combined_signal > 0.3 and ml_confidence > self.ml_confidence_threshold:
            final_signal = 1  # Long
        elif combined_signal < -0.3 and ml_confidence > self.ml_confidence_threshold:
            final_signal = -1  # Short
        else:
            final_signal = 0  # Neutral
        
        # Signal metadata
        signal_metadata = {
            'combined_signal': combined_signal,
            'supertrend_weight': supertrend_weight,
            'ml_weight': ml_weight,
            'ml_confidence': ml_confidence,
            'market_conditions': market_conditions,
            'final_signal': final_signal
        }
        
        return final_signal, combined_signal, signal_metadata
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, confidence: float) -> int:
        """
        Calculate position size based on risk and confidence
        """
        # Base risk calculation
        risk_per_share = abs(entry_price - stop_loss)
        base_risk_amount = self.capital * self.risk_per_trade
        
        # Adjust for confidence
        confidence_multiplier = min(2.0, max(0.5, confidence * 2))
        
        # Calculate final position size
        adjusted_risk_amount = base_risk_amount * confidence_multiplier
        shares = int(adjusted_risk_amount / risk_per_share)
        
        return max(1, shares)
    
    def run_integrated_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run the integrated SuperTrend + ML strategy
        """
        print("🚀 Running Integrated SuperTrend + ML Strategy...")
        
        # Calculate SuperTrend
        df = self.calculate_supertrend(df)
        
        # Train optimized models
        self.train_optimized_models(df)
        
        # Initialize strategy variables
        self.capital = self.initial_capital
        self.trades = []
        self.signal_history = []
        
        # Run strategy
        for i in range(100, len(df)):  # Start after enough data for indicators
            current_data = df.iloc[:i+1]
            current_row = df.iloc[i]
            
            # Get SuperTrend signal
            supertrend_signal = current_row['supertrend_direction']
            
            # Get ML prediction
            features = self.engineer_features(current_data).iloc[-1:]
            feature_columns = [
                'price_change', 'price_change_abs', 'high_low_ratio', 'open_close_ratio',
                'volume_change', 'volume_ratio', 'volatility', 'atr',
                'price_vs_ma5', 'price_vs_ma10', 'price_vs_ma20',
                'hour', 'day_of_week', 'is_market_open'
            ]
            
            if 'price_vs_supertrend' in features.columns:
                feature_columns.extend(['price_vs_supertrend', 'supertrend_distance'])
            
            if len(feature_columns) > 0:
                ml_prediction, ml_confidence = self.get_ensemble_prediction(features[feature_columns])
            else:
                ml_prediction, ml_confidence = 0, 0.5
            
            # Market conditions
            market_conditions = {
                'volatility': current_row.get('volatility', 0.01),
                'volume_ratio': current_row.get('volume_ratio', 1.0)
            }
            
            # Get integrated signal
            signal, signal_strength, metadata = self.integrate_supertrend_ml_signals(
                supertrend_signal, ml_prediction, ml_confidence, market_conditions
            )
            
            # Execute trades
            self.execute_trade(signal, current_row, metadata)
            
            # Record signal
            self.signal_history.append({
                'timestamp': current_row['timestamp'],
                'signal': signal,
                'signal_strength': signal_strength,
                'price': current_row['close'],
                'metadata': metadata
            })
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics()
        
        print("✅ Integrated strategy completed")
        return performance
    
    def execute_trade(self, signal: int, current_row: pd.Series, metadata: Dict[str, Any]):
        """
        Execute trade based on signal
        """
        current_price = current_row['close']
        
        if signal == 1 and self.current_position != 'long':  # Long signal
            if self.current_position == 'short':
                self.close_position(current_price, 'short')
            
            # Open long position
            stop_loss = current_price * 0.98  # 2% stop loss
            position_size = self.calculate_position_size(
                current_price, stop_loss, metadata['ml_confidence']
            )
            
            self.current_position = 'long'
            self.trades.append({
                'timestamp': current_row['timestamp'],
                'action': 'buy',
                'price': current_price,
                'size': position_size,
                'signal_strength': metadata['combined_signal'],
                'metadata': metadata
            })
            
        elif signal == -1 and self.current_position != 'short':  # Short signal
            if self.current_position == 'long':
                self.close_position(current_price, 'long')
            
            # Open short position
            stop_loss = current_price * 1.02  # 2% stop loss
            position_size = self.calculate_position_size(
                current_price, stop_loss, metadata['ml_confidence']
            )
            
            self.current_position = 'short'
            self.trades.append({
                'timestamp': current_row['timestamp'],
                'action': 'sell',
                'price': current_price,
                'size': position_size,
                'signal_strength': metadata['combined_signal'],
                'metadata': metadata
            })
    
    def close_position(self, current_price: float, position_type: str):
        """
        Close current position
        """
        if self.trades:
            last_trade = self.trades[-1]
            if last_trade['action'] in ['buy', 'sell']:
                self.trades.append({
                    'timestamp': last_trade['timestamp'],
                    'action': 'close',
                    'price': current_price,
                    'size': last_trade['size'],
                    'signal_strength': 0,
                    'metadata': {'position_type': position_type}
                })
        
        self.current_position = None
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics
        """
        if not self.trades:
            return {}
        
        # Calculate returns
        returns = []
        capital_curve = [self.initial_capital]
        current_capital = self.initial_capital
        
        for i in range(0, len(self.trades), 2):
            if i + 1 < len(self.trades):
                entry_trade = self.trades[i]
                exit_trade = self.trades[i + 1]
                
                if entry_trade['action'] == 'buy' and exit_trade['action'] == 'close':
                    trade_return = (exit_trade['price'] - entry_trade['price']) / entry_trade['price']
                elif entry_trade['action'] == 'sell' and exit_trade['action'] == 'close':
                    trade_return = (entry_trade['price'] - exit_trade['price']) / entry_trade['price']
                else:
                    continue
                
                returns.append(trade_return)
                current_capital *= (1 + trade_return)
                capital_curve.append(current_capital)
        
        if not returns:
            return {}
        
        # Calculate metrics
        total_return = (current_capital - self.initial_capital) / self.initial_capital
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        max_drawdown = self.calculate_max_drawdown(capital_curve)
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        
        return {
            'total_return': total_return,
            'avg_return': avg_return,
            'std_return': std_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(returns),
            'capital_curve': capital_curve,
            'returns': returns
        }
    
    def calculate_max_drawdown(self, capital_curve: List[float]) -> float:
        """
        Calculate maximum drawdown
        """
        peak = capital_curve[0]
        max_dd = 0
        
        for capital in capital_curve:
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def plot_performance(self, performance: Dict[str, Any]):
        """
        Plot strategy performance
        """
        if 'capital_curve' not in performance:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Capital curve
        ax1.plot(performance['capital_curve'])
        ax1.set_title('Capital Curve')
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('Capital ($)')
        ax1.grid(True)
        
        # Returns distribution
        if 'returns' in performance:
            ax2.hist(performance['returns'], bins=20, alpha=0.7)
            ax2.set_title('Returns Distribution')
            ax2.set_xlabel('Return')
            ax2.set_ylabel('Frequency')
            ax2.grid(True)
        
        # Model performance comparison
        model_names = list(self.performance_metrics.keys())
        accuracies = [self.performance_metrics[name]['accuracy'] for name in model_names]
        aucs = [self.performance_metrics[name]['auc'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax3.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.7)
        ax3.bar(x + width/2, aucs, width, label='AUC', alpha=0.7)
        ax3.set_title('Model Performance Comparison')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names)
        ax3.legend()
        ax3.grid(True)
        
        # Signal strength distribution
        signal_strengths = [s['signal_strength'] for s in self.signal_history]
        ax4.hist(signal_strengths, bins=20, alpha=0.7)
        ax4.set_title('Signal Strength Distribution')
        ax4.set_xlabel('Signal Strength')
        ax4.set_ylabel('Frequency')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, filepath: str):
        """
        Save trained models
        """
        model_data = {
            'models': self.models,
            'best_params': self.best_params,
            'performance_metrics': self.performance_metrics
        }
        joblib.dump(model_data, filepath)
        print(f"✅ Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """
        Load trained models
        """
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.best_params = model_data['best_params']
        self.performance_metrics = model_data['performance_metrics']
        print(f"✅ Models loaded from {filepath}")


def main():
    """
    Main function to run the integrated strategy
    """
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv('data/cache_SOXL_10Min.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Initialize strategy
    strategy = AdvancedSuperTrendMLIntegration(
        initial_capital=100000,
        risk_per_trade=0.02,
        ml_confidence_threshold=0.65,
        supertrend_weight=0.6,
        ml_weight=0.4
    )
    
    # Run strategy
    performance = strategy.run_integrated_strategy(df)
    
    # Print results
    print("\n📈 Strategy Performance:")
    for metric, value in performance.items():
        if metric not in ['capital_curve', 'returns']:
            print(f"{metric}: {value:.4f}")
    
    # Plot performance
    strategy.plot_performance(performance)
    
    # Save models
    strategy.save_models('models/integrated_supertrend_ml_models.pkl')
    
    print("\n✅ Advanced SuperTrend + ML Integration completed!")


if __name__ == "__main__":
    main() 