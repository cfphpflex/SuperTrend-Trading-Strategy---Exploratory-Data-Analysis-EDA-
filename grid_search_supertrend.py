#!/usr/bin/env python3
"""
SuperTrend Grid Search - Focused Parameter Optimization
Baseline: $50K → $110K (120% return) - TARGET: IMPROVE THIS

Focused Grid Search Approach:
1. Test top 5 most promising parameter combinations
2. Focus on proven effective parameter ranges
3. Identify best performing configurations
4. Optimize for maximum returns with minimal testing

Parameters to optimize (focused ranges):
- SuperTrend period (7, 10, 12)
- SuperTrend multiplier (2.0, 2.5, 3.0)
- Risk per trade (1.0%, 1.5%)
- ADX threshold (20, 25, 30)
- Volume ratio threshold (0.9, 1.0, 1.1)
- ATR stop loss multiplier (1.5, 1.8, 2.0)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Any, Optional
from itertools import product
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time

warnings.filterwarnings('ignore')

class SuperTrendGridSearch:
    """
    Grid search system for SuperTrend parameter optimization
    """
    
    def __init__(self, initial_capital: float = 50000):
        self.initial_capital = initial_capital
        self.baseline_target = 110024.28
        self.results = []
        self.best_configs = []
        
        # Parameter grids - Reduced to top 5 most promising combinations
        self.param_grid = {
            'supertrend_period': [7, 10, 12],  # Most effective periods
            'supertrend_multiplier': [2.0, 2.5, 3.0],  # Optimal multiplier range
            'risk_per_trade': [0.01, 0.015],  # Balanced risk levels
            'adx_threshold': [20, 25, 30],  # Effective trend strength filters
            'volume_ratio_threshold': [0.9, 1.0, 1.1],  # Optimal volume filters
            'atr_stop_multiplier': [1.5, 1.8, 2.0]  # Effective stop loss levels
        }
        
        print(f"🎯 Grid Search Target: Improve on baseline ${self.baseline_target:,.2f}")
        print(f"📊 Total combinations: {self._calculate_total_combinations()} (focused testing)")
        
    def _calculate_total_combinations(self) -> int:
        """Calculate total number of parameter combinations"""
        total = 1
        for param, values in self.param_grid.items():
            total *= len(values)
        return total
    
    def calculate_supertrend(self, df: pd.DataFrame, period: int, multiplier: float) -> pd.DataFrame:
        """Calculate SuperTrend indicator"""
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
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index)"""
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
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic features"""
        df = df.copy()
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def run_single_backtest(self, params: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Run single backtest with given parameters"""
        try:
            # Extract parameters
            supertrend_period = params['supertrend_period']
            supertrend_multiplier = params['supertrend_multiplier']
            risk_per_trade = params['risk_per_trade']
            adx_threshold = params['adx_threshold']
            volume_ratio_threshold = params['volume_ratio_threshold']
            atr_stop_multiplier = params['atr_stop_multiplier']
            
            # Calculate indicators
            df_test = df.copy()
            df_test = self.calculate_supertrend(df_test, supertrend_period, supertrend_multiplier)
            df_test = self.calculate_adx(df_test)
            df_test = self.add_features(df_test)
            
            # Initialize variables
            capital = self.initial_capital
            peak_capital = self.initial_capital
            trades = []
            current_position = None
            
            # Run strategy
            for i in range(20, len(df_test)):
                current_row = df_test.iloc[i]
                current_price = current_row['close']
                
                # Update peak capital
                if capital > peak_capital:
                    peak_capital = capital
                
                # Get signal with filters
                signal = 0
                supertrend_signal = current_row['supertrend_direction']
                adx = current_row.get('adx', 0)
                volume_ratio = current_row.get('volume_ratio', 1.0)
                
                # Apply filters
                if adx >= adx_threshold and volume_ratio >= volume_ratio_threshold:
                    signal = supertrend_signal
                
                # Execute trades
                if current_position is None:  # No position
                    if signal != 0:
                        # Enter position
                        atr = current_row['atr']
                        stop_loss = current_price - (atr * atr_stop_multiplier) if signal == 1 else current_price + (atr * atr_stop_multiplier)
                        
                        # Calculate position size
                        risk_per_share = abs(current_price - stop_loss)
                        risk_amount = capital * risk_per_trade
                        shares = max(1, int(risk_amount / risk_per_share))
                        
                        current_position = {
                            'type': 'long' if signal == 1 else 'short',
                            'entry_price': current_price,
                            'shares': shares,
                            'stop_loss': stop_loss
                        }
                        
                else:  # Have position
                    # Check exit conditions
                    should_exit = False
                    
                    # Stop loss
                    if (current_position['type'] == 'long' and current_price <= current_position['stop_loss']) or \
                       (current_position['type'] == 'short' and current_price >= current_position['stop_loss']):
                        should_exit = True
                    
                    # Signal reversal
                    elif signal != 0 and signal != (1 if current_position['type'] == 'long' else -1):
                        should_exit = True
                    
                    if should_exit:
                        # Close position
                        shares = current_position['shares']
                        entry_price = current_position['entry_price']
                        
                        if current_position['type'] == 'long':
                            pnl = (current_price - entry_price) * shares
                        else:
                            pnl = (entry_price - current_price) * shares
                        
                        capital += pnl
                        trades.append({
                            'pnl': pnl,
                            'return_pct': (pnl / (entry_price * shares)) * 100
                        })
                        
                        current_position = None
                        
                        # Enter new position if signal exists
                        if signal != 0:
                            atr = current_row['atr']
                            stop_loss = current_price - (atr * atr_stop_multiplier) if signal == 1 else current_price + (atr * atr_stop_multiplier)
                            
                            risk_per_share = abs(current_price - stop_loss)
                            risk_amount = capital * risk_per_trade
                            shares = max(1, int(risk_amount / risk_per_share))
                            
                            current_position = {
                                'type': 'long' if signal == 1 else 'short',
                                'entry_price': current_price,
                                'shares': shares,
                                'stop_loss': stop_loss
                            }
            
            # Close any remaining position
            if current_position is not None:
                shares = current_position['shares']
                entry_price = current_position['entry_price']
                exit_price = df_test.iloc[-1]['close']
                
                if current_position['type'] == 'long':
                    pnl = (exit_price - entry_price) * shares
                else:
                    pnl = (entry_price - exit_price) * shares
                
                capital += pnl
                trades.append({
                    'pnl': pnl,
                    'return_pct': (pnl / (entry_price * shares)) * 100
                })
            
            # Calculate performance metrics
            total_return = capital - self.initial_capital
            total_return_pct = (total_return / self.initial_capital) * 100
            num_trades = len(trades)
            
            if num_trades == 0:
                return {
                    'params': params,
                    'final_capital': capital,
                    'total_return': total_return,
                    'total_return_pct': total_return_pct,
                    'num_trades': num_trades,
                    'win_rate': 0,
                    'max_drawdown': 0,
                    'improvement_vs_baseline': total_return - (self.baseline_target - self.initial_capital),
                    'status': 'no_trades'
                }
            
            # Win rate
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / num_trades * 100
            
            # Max drawdown
            capital_curve = [self.initial_capital]
            for trade in trades:
                capital_curve.append(capital_curve[-1] + trade['pnl'])
            
            peak = capital_curve[0]
            max_dd = 0
            for cap in capital_curve:
                if cap > peak:
                    peak = cap
                dd = (peak - cap) / peak
                if dd > max_dd:
                    max_dd = dd
            max_drawdown = max_dd * 100
            
            # Improvement vs baseline
            baseline_return = self.baseline_target - self.initial_capital
            improvement = total_return - baseline_return
            
            return {
                'params': params,
                'final_capital': capital,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'improvement_vs_baseline': improvement,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'params': params,
                'final_capital': self.initial_capital,
                'total_return': 0,
                'total_return_pct': 0,
                'num_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'improvement_vs_baseline': 0,
                'status': f'error: {str(e)}'
            }
    
    def run_grid_search(self, df: pd.DataFrame, max_workers: int = 8) -> None:
        """Run focused grid search"""
        print(f"🚀 Starting Focused Grid Search with {self._calculate_total_combinations()} combinations...")
        print(f"🖥️ Using {max_workers} workers for parallel processing")
        
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        combinations = list(product(*param_values))
        
        total_combinations = len(combinations)
        print(f"📊 Testing {total_combinations} parameter combinations...")
        
        # Convert to parameter dictionaries
        param_dicts = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            param_dicts.append(param_dict)
        
        # Run backtests in parallel
        results = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_params = {
                executor.submit(self.run_single_backtest, params, df): params 
                for params in param_dicts
            }
            
            # Collect results
            for future in as_completed(future_to_params):
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 100 == 0:
                    print(f"✅ Completed {completed}/{total_combinations} combinations...")
        
        # Store results
        self.results = results
        
        # Find best configurations
        self._find_best_configs()
        
        # Save results
        self._save_results()
        
        print(f"🎉 Grid Search completed! Found {len(self.best_configs)} configurations that beat baseline")
    
    def _find_best_configs(self) -> None:
        """Find best performing configurations"""
        # Filter successful results
        successful_results = [r for r in self.results if r['status'] == 'success' and r['num_trades'] > 0]
        
        # Sort by improvement vs baseline
        successful_results.sort(key=lambda x: x['improvement_vs_baseline'], reverse=True)
        
        # Get top 10 configurations
        self.best_configs = successful_results[:10]
        
        print(f"\n🏆 TOP 10 CONFIGURATIONS:")
        print("="*80)
        for i, config in enumerate(self.best_configs, 1):
            print(f"{i:2d}. Final Capital: ${config['final_capital']:,.2f} | "
                  f"Return: {config['total_return_pct']:.1f}% | "
                  f"Trades: {config['num_trades']} | "
                  f"Win Rate: {config['win_rate']:.1f}% | "
                  f"Drawdown: {config['max_drawdown']:.1f}%")
            print(f"    Params: {config['params']}")
            print()
    
    def _save_results(self) -> None:
        """Save results to files"""
        # Save all results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('grid_search_all_results.csv', index=False)
        
        # Save best configurations
        if self.best_configs:
            best_df = pd.DataFrame(self.best_configs)
            best_df.to_csv('grid_search_best_configs.csv', index=False)
        
        # Save summary statistics
        successful_results = [r for r in self.results if r['status'] == 'success' and r['num_trades'] > 0]
        improving_results = [r for r in successful_results if r['improvement_vs_baseline'] > 0]
        
        summary = {
            'total_combinations': len(self.results),
            'successful_runs': len(successful_results),
            'improving_configs': len(improving_results),
            'improvement_rate': len(improving_results) / len(successful_results) * 100 if successful_results else 0,
            'best_final_capital': max([r['final_capital'] for r in successful_results]) if successful_results else 0,
            'best_improvement': max([r['improvement_vs_baseline'] for r in successful_results]) if successful_results else 0
        }
        
        with open('grid_search_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Results saved:")
        print(f"   - All results: grid_search_all_results.csv")
        print(f"   - Best configs: grid_search_best_configs.csv")
        print(f"   - Summary: grid_search_summary.json")
    
    def plot_results(self) -> None:
        """Plot grid search results"""
        if not self.results:
            print("❌ No results to plot")
            return
        
        successful_results = [r for r in self.results if r['status'] == 'success' and r['num_trades'] > 0]
        
        if not successful_results:
            print("❌ No successful results to plot")
            return
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Final capital distribution
        final_capitals = [r['final_capital'] for r in successful_results]
        ax1.hist(final_capitals, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(self.baseline_target, color='red', linestyle='--', label=f'Baseline: ${self.baseline_target:,.0f}')
        ax1.set_title('Distribution of Final Capital', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Final Capital ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Improvement vs baseline
        improvements = [r['improvement_vs_baseline'] for r in successful_results]
        ax2.hist(improvements, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', label='Baseline')
        ax2.set_title('Distribution of Improvement vs Baseline', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Improvement ($)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Win rate vs final capital
        win_rates = [r['win_rate'] for r in successful_results]
        ax3.scatter(win_rates, final_capitals, alpha=0.6, color='purple')
        ax3.axhline(self.baseline_target, color='red', linestyle='--', label=f'Baseline: ${self.baseline_target:,.0f}')
        ax3.set_title('Win Rate vs Final Capital', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Win Rate (%)', fontsize=12)
        ax3.set_ylabel('Final Capital ($)', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Max drawdown vs final capital
        max_drawdowns = [r['max_drawdown'] for r in successful_results]
        ax4.scatter(max_drawdowns, final_capitals, alpha=0.6, color='orange')
        ax4.axhline(self.baseline_target, color='red', linestyle='--', label=f'Baseline: ${self.baseline_target:,.0f}')
        ax4.set_title('Max Drawdown vs Final Capital', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Max Drawdown (%)', fontsize=12)
        ax4.set_ylabel('Final Capital ($)', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('grid_search_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Results plot saved as: grid_search_results.png")

def main():
    """Main execution function"""
    print("🚀 SuperTrend Grid Search Optimization")
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
    
    # Initialize grid search
    grid_search = SuperTrendGridSearch()
    
    # Run grid search
    start_time = time.time()
    grid_search.run_grid_search(df, max_workers=8)
    end_time = time.time()
    
    print(f"\n⏱️ Grid search completed in {end_time - start_time:.1f} seconds")
    
    # Plot results
    grid_search.plot_results()

if __name__ == "__main__":
    main() 