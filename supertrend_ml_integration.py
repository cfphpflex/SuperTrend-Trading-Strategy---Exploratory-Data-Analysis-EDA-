#!/usr/bin/env python3
"""
Baseline SuperTrend Strategy - Reverting to Working Version
Baseline: $50K → $110K (120% return) - MUST IMPROVE THIS

Simple, proven improvements only:
1. ATR-based dynamic stop loss (instead of fixed 1%)
2. Volume confirmation for entry
3. Risk-adjusted position sizing during drawdown

Goal: Increase final capital above $110,024.28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings('ignore')

class BaselineSuperTrendStrategy:
    """
    Baseline SuperTrend strategy - working version
    Target: >$110,024.28 final capital
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
        
        # Performance tracking
        self.trades = []
        self.signal_history = []
        self.current_position = None
        self.peak_capital = initial_capital
        
        print(f"🎯 Target: Improve on baseline $110,024.28 final capital")
        print(f"🛡️ Risk per trade: {risk_per_trade*100}%")
        
    def calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SuperTrend indicator (baseline method)"""
        df = df.copy()
        
        # Calculate True Range (TR)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=self.supertrend_period).mean()
        
        # Calculate Basic Upper and Lower Bands
        df['basic_upper'] = (df['high'] + df['low']) / 2 + (self.supertrend_multiplier * df['atr'])
        df['basic_lower'] = (df['high'] + df['low']) / 2 - (self.supertrend_multiplier * df['atr'])
        
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
    
    def add_simple_improvements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple, proven improvements"""
        df = df.copy()
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # ATR for dynamic stop loss
        df['atr_5'] = df['atr'].rolling(window=5).mean()
        
        return df
    
    def get_signal(self, df: pd.DataFrame, i: int) -> Tuple[int, float]:
        """Get trading signal with simple improvements"""
        if i < 20:  # Need enough data
            return 0, 0.0
        
        current_row = df.iloc[i]
        supertrend_signal = current_row['supertrend_direction']
        
        # Base signal strength
        signal_strength = 0.0
        
        # SuperTrend signal
        if supertrend_signal == 1:
            signal_strength += 0.5
        elif supertrend_signal == -1:
            signal_strength -= 0.5
        
        # Volume confirmation (simple)
        volume_ratio = current_row.get('volume_ratio', 1.0)
        if volume_ratio > 1.1:  # Above average volume
            signal_strength += 0.1
        elif volume_ratio < 0.9:  # Below average volume
            signal_strength -= 0.1
        
        # Determine final signal
        if signal_strength > 0.3:
            final_signal = 1  # Long
        elif signal_strength < -0.3:
            final_signal = -1  # Short
        else:
            final_signal = 0  # Neutral
        
        return final_signal, signal_strength
    
    def calculate_dynamic_stop_loss(self, entry_price: float, direction: int, atr: float) -> float:
        """Calculate dynamic stop loss based on ATR"""
        # Use 1.5x ATR for stop loss distance (improvement over fixed 1%)
        stop_distance = atr * 1.5
        
        if direction == 1:  # Long
            stop_loss = entry_price - stop_distance
        else:  # Short
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size with risk adjustment"""
        # Base position size
        risk_per_share = abs(entry_price - stop_loss)
        base_risk_amount = self.capital * self.risk_per_trade
        base_shares = int(base_risk_amount / risk_per_share)
        
        # Risk adjustment based on current drawdown
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if current_drawdown > 0.1:  # If drawdown > 10%
            risk_multiplier = 0.7  # Reduce position size
        elif current_drawdown > 0.05:  # If drawdown > 5%
            risk_multiplier = 0.85
        else:
            risk_multiplier = 1.0
        
        adjusted_shares = int(base_shares * risk_multiplier)
        return max(1, adjusted_shares)
    
    def run_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run the baseline strategy with simple improvements"""
        print("🚀 Running Baseline SuperTrend Strategy with Simple Improvements...")
        
        # Calculate SuperTrend
        df = self.calculate_supertrend(df)
        
        # Add simple improvements
        df = self.add_simple_improvements(df)
        
        # Initialize strategy variables
        self.capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.trades = []
        self.signal_history = []
        self.current_position = None
        
        # Run strategy
        for i in range(20, len(df)):
            current_row = df.iloc[i]
            
            # Get signal
            signal, signal_strength = self.get_signal(df, i)
            
            # Execute trades
            self.execute_trade(signal, current_row, signal_strength)
            
            # Update peak capital
            if self.capital > self.peak_capital:
                self.peak_capital = self.capital
            
            # Record signal
            self.signal_history.append({
                'timestamp': current_row['timestamp'],
                'signal': signal,
                'signal_strength': signal_strength,
                'price': current_row['close'],
                'capital': self.capital
            })
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics()
        
        print("✅ Baseline strategy completed")
        return performance
    
    def execute_trade(self, signal: int, current_row: pd.Series, signal_strength: float):
        """Execute trade"""
        current_price = current_row['close']
        
        if signal == 1 and self.current_position != 'long':  # Long signal
            if self.current_position == 'short':
                self.close_position(current_price, 'short')
            
            # Calculate dynamic stop loss
            atr = current_row.get('atr_5', current_row.get('atr', 0.01))
            stop_loss = self.calculate_dynamic_stop_loss(current_price, 1, atr)
            
            # Calculate position size
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            self.current_position = 'long'
            self.trades.append({
                'timestamp': current_row['timestamp'],
                'action': 'buy',
                'price': current_price,
                'size': position_size,
                'signal_strength': signal_strength,
                'stop_loss': stop_loss
            })
            
        elif signal == -1 and self.current_position != 'short':  # Short signal
            if self.current_position == 'long':
                self.close_position(current_price, 'long')
            
            # Calculate dynamic stop loss
            atr = current_row.get('atr_5', current_row.get('atr', 0.01))
            stop_loss = self.calculate_dynamic_stop_loss(current_price, -1, atr)
            
            # Calculate position size
            position_size = self.calculate_position_size(current_price, stop_loss)
            
            self.current_position = 'short'
            self.trades.append({
                'timestamp': current_row['timestamp'],
                'action': 'sell',
                'price': current_price,
                'size': position_size,
                'signal_strength': signal_strength,
                'stop_loss': stop_loss
            })
    
    def close_position(self, current_price: float, position_type: str):
        """Close current position"""
        if self.trades:
            last_trade = self.trades[-1]
            if last_trade['action'] in ['buy', 'sell']:
                # Calculate PnL
                if last_trade['action'] == 'buy':  # Long position
                    pnl = (current_price - last_trade['price']) * last_trade['size']
                else:  # Short position
                    pnl = (last_trade['price'] - current_price) * last_trade['size']
                
                self.trades.append({
                    'timestamp': last_trade['timestamp'],
                    'action': 'close',
                    'price': current_price,
                    'size': last_trade['size'],
                    'pnl': pnl,
                    'position_type': position_type
                })
        
        self.current_position = None
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
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
        
        # Compare to baseline
        baseline_final_capital = 110024.28
        improvement = current_capital - baseline_final_capital
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': current_capital,
            'total_return': total_return,
            'avg_return': avg_return,
            'std_return': std_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(returns),
            'capital_curve': capital_curve,
            'returns': returns,
            'baseline_comparison': {
                'baseline_final_capital': baseline_final_capital,
                'improvement': improvement,
                'improvement_pct': (improvement / baseline_final_capital) * 100
            }
        }
    
    def calculate_max_drawdown(self, capital_curve: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = capital_curve[0]
        max_dd = 0
        
        for capital in capital_curve:
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def print_performance_summary(self, performance: Dict[str, Any]):
        """Print performance summary with baseline comparison"""
        if not performance:
            print("❌ No performance data available")
            return
            
        print("\n" + "="*60)
        print("📊 BASELINE PERFORMANCE SUMMARY")
        print("="*60)
        
        # Safe access to performance metrics
        initial_capital = performance.get('initial_capital', self.initial_capital)
        final_capital = performance.get('final_capital', 0)
        total_return = performance.get('total_return', 0)
        num_trades = performance.get('num_trades', 0)
        win_rate = performance.get('win_rate', 0)
        avg_return = performance.get('avg_return', 0)
        max_drawdown = performance.get('max_drawdown', 0)
        sharpe_ratio = performance.get('sharpe_ratio', 0)
        
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Capital: ${final_capital:,.2f}")
        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Number of Trades: {num_trades}")
        print(f"Win Rate: {win_rate*100:.1f}%")
        print(f"Average Trade: ${avg_return*initial_capital:,.2f}")
        print(f"Max Drawdown: ${max_drawdown*initial_capital:,.2f} ({max_drawdown*100:.1f}%)")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        
        # Baseline comparison
        baseline = performance.get('baseline_comparison', {})
        if baseline:
            baseline_final_capital = baseline.get('baseline_final_capital', 110024.28)
            improvement = baseline.get('improvement', final_capital - 110024.28)
            improvement_pct = baseline.get('improvement_pct', (improvement / 110024.28) * 100)
            
            print(f"\n🎯 BASELINE COMPARISON:")
            print(f"Baseline Final Capital: ${baseline_final_capital:,.2f}")
            print(f"Improvement: ${improvement:,.2f} ({improvement_pct:+.2f}%)")
            
            if improvement > 0:
                print("✅ IMPROVEMENT SUCCESSFUL - Above baseline!")
            else:
                print("❌ IMPROVEMENT FAILED - Below baseline performance")
        
        print("="*60)


def main():
    """Main function"""
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv('data/cache_SOXL_10Min.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Initialize strategy
    strategy = BaselineSuperTrendStrategy(
        initial_capital=50000,
        risk_per_trade=0.01
    )
    
    # Run strategy
    performance = strategy.run_strategy(df)
    
    # Print performance summary
    strategy.print_performance_summary(performance)
    
    # Plot performance
    if 'capital_curve' in performance:
        plt.figure(figsize=(12, 6))
        plt.plot(performance['capital_curve'])
        plt.axhline(y=110024.28, color='r', linestyle='--', label='Baseline Target')
        plt.title('Baseline SuperTrend Strategy - Capital Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    print("\n✅ Baseline SuperTrend Strategy completed!")


if __name__ == "__main__":
    main() 