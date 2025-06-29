#!/usr/bin/env python3
"""
Conservative SuperTrend Improvement Strategy
Baseline: $50K → $110K (120% return) - TARGET: IMPROVE THIS

SIMPLE, PROVEN IMPROVEMENTS ONLY:
1. ATR-based dynamic stop loss (instead of fixed 1%)
2. Volume confirmation for entry (simple filter)
3. Risk-adjusted position sizing during drawdown
4. Basic trend strength filter (ADX > 25)

NO COMPLEX ML - ONLY PROVEN TECHNICAL INDICATORS
Goal: Increase final capital above $110,024.28 with reduced drawdown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings('ignore')

class ConservativeSuperTrendStrategy:
    """
    Conservative SuperTrend strategy with simple, proven improvements
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
        
        # Performance tracking
        self.trades = []
        self.signal_history = []
        self.current_position = None
        self.peak_capital = initial_capital
        
        print(f"🎯 Target: Improve on baseline $110,024.28 final capital")
        print(f"🛡️ Conservative approach: Simple, proven improvements only")
        
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
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index) for trend strength filter"""
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
    
    def add_simple_improvements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple, proven improvements"""
        df = df.copy()
        
        # Volume confirmation
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # ATR for dynamic stop loss
        df['atr_5'] = df['atr'].rolling(window=5).mean()
        
        return df
    
    def get_conservative_signal(self, df: pd.DataFrame, i: int) -> Tuple[int, float]:
        """Get conservative trading signal with simple filters"""
        if i < 20:  # Need enough data
            return 0, 0.0
        
        current_row = df.iloc[i]
        supertrend_signal = current_row['supertrend_direction']
        
        # 1. ADX Trend Strength Filter - Only trade strong trends
        adx = current_row.get('adx', 0)
        if adx < 25:  # Weak trend - skip trade
            return 0, 0.0
        
        # 2. Volume Confirmation - Simple filter
        volume_ratio = current_row.get('volume_ratio', 1.0)
        if volume_ratio < 0.8:  # Below average volume - skip trade
            return 0, 0.0
        
        # Base signal strength
        signal_strength = 0.0
        
        # SuperTrend signal
        if supertrend_signal == 1:
            signal_strength += 0.6
        elif supertrend_signal == -1:
            signal_strength -= 0.6
        
        # Volume boost
        if volume_ratio > 1.2:  # Strong volume
            signal_strength += 0.1
        elif volume_ratio > 1.0:  # Above average volume
            signal_strength += 0.05
        
        # ADX strength boost
        if adx > 40:  # Very strong trend
            signal_strength += 0.1
        elif adx > 30:  # Strong trend
            signal_strength += 0.05
        
        # Determine final signal with higher threshold
        if signal_strength > 0.5:
            final_signal = 1  # Long
        elif signal_strength < -0.5:
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
        """Calculate position size with conservative risk management"""
        # Base position size
        risk_per_share = abs(entry_price - stop_loss)
        base_risk_amount = self.capital * self.risk_per_trade
        base_shares = int(base_risk_amount / risk_per_share)
        
        # Conservative risk adjustment based on current drawdown
        current_drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if current_drawdown > 0.15:  # If drawdown > 15%
            risk_multiplier = 0.5  # Reduce position size significantly
        elif current_drawdown > 0.10:  # If drawdown > 10%
            risk_multiplier = 0.7  # Reduce position size
        elif current_drawdown > 0.05:  # If drawdown > 5%
            risk_multiplier = 0.85
        else:
            risk_multiplier = 1.0
        
        adjusted_shares = int(base_shares * risk_multiplier)
        return max(1, adjusted_shares)  # Minimum 1 share
    
    def run_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run the conservative strategy"""
        print("🚀 Running Conservative SuperTrend Strategy...")
        
        # Calculate SuperTrend
        df = self.calculate_supertrend(df)
        
        # Calculate ADX for trend strength filter
        df = self.calculate_adx(df)
        
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
            current_price = current_row['close']
            
            # Update peak capital
            if self.capital > self.peak_capital:
                self.peak_capital = self.capital
            
            # Get conservative signal
            signal, signal_strength = self.get_conservative_signal(df, i)
            
            # Execute trades
            if self.current_position is None:  # No position
                if signal != 0:
                    self.execute_trade(signal, current_row, signal_strength)
            else:  # Have position
                # Check stop loss or take profit
                if self.check_exit_conditions(current_row):
                    self.close_position(current_price, self.current_position['type'])
                # Check for signal reversal
                elif signal != 0 and signal != self.current_position['type']:
                    self.close_position(current_price, self.current_position['type'])
                    self.execute_trade(signal, current_row, signal_strength)
            
            # Record signal
            self.signal_history.append({
                'date': current_row.name,
                'price': current_price,
                'signal': signal,
                'signal_strength': signal_strength,
                'capital': self.capital
            })
        
        # Close any remaining position
        if self.current_position is not None:
            self.close_position(df.iloc[-1]['close'], self.current_position['type'])
        
        # Calculate performance
        performance = self.calculate_performance_metrics()
        self.print_performance_summary(performance)
        
        return performance
    
    def execute_trade(self, signal: int, current_row: pd.Series, signal_strength: float):
        """Execute trade with conservative approach"""
        entry_price = current_row['close']
        atr = current_row['atr']
        
        # Calculate dynamic stop loss
        stop_loss = self.calculate_dynamic_stop_loss(entry_price, signal, atr)
        
        # Calculate position size with conservative risk management
        shares = self.calculate_position_size(entry_price, stop_loss)
        
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
            'signal_strength': signal_strength
        }
        
        # Update capital
        self.capital -= trade_value
        
        print(f"📈 {position_type.upper()} Entry: {shares} shares @ ${entry_price:.2f}")
        print(f"   Stop Loss: ${stop_loss:.2f}, Signal Strength: {signal_strength:.2f}")
        print(f"   Capital: ${self.capital:.2f}")
    
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
        print("🎯 CONSERVATIVE STRATEGY PERFORMANCE SUMMARY")
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
    print("🚀 Conservative SuperTrend Strategy")
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
    
    # Run conservative strategy
    strategy = ConservativeSuperTrendStrategy()
    performance = strategy.run_strategy(df)
    
    # Save results
    results_df = pd.DataFrame(strategy.trades)
    if not results_df.empty:
        results_df.to_csv('conservative_results.csv', index=False)
        print(f"✅ Results saved to conservative_results.csv")

if __name__ == "__main__":
    main() 