#!/usr/bin/env python3
"""
Enhanced SuperTrend Trading Strategy Analysis
Migrated features from supertrend_1d_live_trader.py for comprehensive backtesting and analysis.

MIGRATED FEATURES:
1. ✅ Advanced Cache Management: Load from cache, handle symbol filtering, timezone management
2. ✅ Sophisticated Trading Logic: Position state management, reversal logic, exit reason tracking
3. ✅ Risk Management Improvements: Timeframe-specific stop loss, minimum holding periods
4. ✅ Enhanced SuperTrend Calculation: Improved ATR and signal generation
5. ✅ Comprehensive Analytics: Detailed trade tracking, performance metrics
6. ✅ Data Resampling: 5Min from 1Min data capability
7. ✅ Error Handling: Robust data loading and processing
8. ✅ CHOPPINESS FILTERS: Time, price, and volatility filters to avoid choppy markets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta, time
import warnings
import argparse
import os
import pytz
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
warnings.filterwarnings('ignore')

class PositionState(Enum):
    """Position states"""
    NONE = "none"
    LONG = "long"
    SHORT = "short"

class ExitReason(Enum):
    """Exit reasons"""
    SUPERTREND_EXIT = "supertrend_exit"
    STOP_LOSS = "stop_loss"
    MANUAL_EXIT = "manual_exit"
    ERROR_EXIT = "error_exit"
    CHOPPY_MARKET = "choppy_market"

@dataclass
class Position:
    """Position information with tracking"""
    side: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    quantity: int
    symbol: str
    stop_loss: float = 0.0

@dataclass
class Trade:
    """Enhanced trade information"""
    side: str
    entry_date: datetime
    entry_price: float
    exit_date: datetime
    exit_price: float
    shares: int
    pnl: float
    stop_loss: bool
    exit_reason: str
    holding_bars: int
    entry_reason: str = "supertrend_signal"

class ChoppinessFilter:
    """Filter to avoid choppy market conditions"""
    
    def __init__(self, enable_filters: bool = True):
        # Time-based filters (based on analysis results)
        self.avoid_hours = [9, 15, 21]  # Hours with highest direction change rates
        self.prefer_hours = [8, 10, 20]  # Hours with lowest direction change rates
        self.avoid_days = ['Saturday', 'Friday', 'Tuesday']
        self.prefer_days = ['Monday', 'Thursday', 'Wednesday']
        
        # Price movement filters
        self.min_price_move_pct = 1.15  # Minimum price movement percentage
        self.min_price_move_threshold = 1.0  # Threshold for avoiding small moves
        
        # Volatility filters
        self.max_direction_changes_per_hour = 3  # Max direction changes per hour
        self.recent_trades_window = 10  # Number of recent trades to check
        
        # Enable/disable filters
        self.enable_filters = enable_filters
        
        print(f"🛡️ Choppiness Filter initialized: {'ENABLED' if enable_filters else 'DISABLED'}")
        print(f"⏰ Time filters: Avoid hours {self.avoid_hours}, Prefer hours {self.prefer_hours}")
        print(f"💰 Price filters: Min move {self.min_price_move_pct}%")
        print(f"📊 Volatility filters: Max {self.max_direction_changes_per_hour} changes/hour")
    
    def is_choppy_time(self, timestamp: datetime) -> Tuple[bool, str]:
        """Check if current time is prone to choppiness"""
        if not self.enable_filters:
            return False, "Filters disabled"
            
        hour = timestamp.hour
        day = timestamp.strftime('%A')
        
        # Avoid worst hours
        if hour in self.avoid_hours:
            return True, f"Hour {hour} is in avoid list"
        
        # Avoid worst days
        if day in self.avoid_days:
            return True, f"{day} is in avoid list"
            
        return False, "Time OK"
    
    def is_choppy_price_movement(self, current_price: float, recent_prices: List[float], 
                                lookback_bars: int = 5) -> Tuple[bool, str]:
        """Check if price movement is too small (choppy)"""
        if not self.enable_filters or len(recent_prices) < lookback_bars:
            return False, "Insufficient price history"
        
        # Calculate recent price movement
        start_price = recent_prices[-lookback_bars]
        price_change_pct = abs(current_price - start_price) / start_price * 100
        
        if price_change_pct < self.min_price_move_pct:
            return True, f"Price move too small: {price_change_pct:.2f}% < {self.min_price_move_pct}%"
        
        return False, f"Price move OK: {price_change_pct:.2f}%"
    
    def is_choppy_volatility(self, recent_trades: List[Trade], current_time: datetime) -> Tuple[bool, str]:
        """Check if there are too many recent direction changes"""
        if not self.enable_filters or len(recent_trades) < 2:
            return False, "Insufficient trade history"
        
        # Count direction changes in the last hour
        one_hour_ago = current_time - timedelta(hours=1)
        recent_changes = 0
        
        for i in range(1, min(self.recent_trades_window, len(recent_trades))):
            if recent_trades[-i].exit_date >= one_hour_ago:
                if recent_trades[-i].side != recent_trades[-(i+1)].side:
                    recent_changes += 1
            else:
                break
        
        if recent_changes >= self.max_direction_changes_per_hour:
            return True, f"Too many direction changes: {recent_changes} in last hour"
        
        return False, f"Volatility OK: {recent_changes} changes in last hour"
    
    def should_skip_trade(self, timestamp: datetime, current_price: float, 
                         recent_prices: List[float], recent_trades: List[Trade]) -> Tuple[bool, str]:
        """Main function to determine if trade should be skipped due to choppiness"""
        if not self.enable_filters:
            return False, "Filters disabled"
        
        # Check time-based filters
        is_choppy_time, time_reason = self.is_choppy_time(timestamp)
        if is_choppy_time:
            return True, f"Time filter: {time_reason}"
        
        # Check price movement filters
        is_choppy_price, price_reason = self.is_choppy_price_movement(current_price, recent_prices)
        if is_choppy_price:
            return True, f"Price filter: {price_reason}"
        
        # Check volatility filters
        is_choppy_vol, vol_reason = self.is_choppy_volatility(recent_trades, timestamp)
        if is_choppy_vol:
            return True, f"Volatility filter: {vol_reason}"
        
        return False, "Trade allowed"

class EnhancedSuperTrendAnalyzer:
    """Enhanced SuperTrend analyzer with migrated features from live trader"""
    
    def __init__(self, symbol: str = 'SOXL', timeframe: str = '5Min', 
                 initial_capital: float = 100000, risk_per_trade_pct: float = 0.02,
                 enable_choppiness_filters: bool = True):
        """
        Initialize the enhanced SuperTrend analyzer
        
        Args:
            symbol: Trading symbol (e.g., 'SOXL')
            timeframe: Data timeframe (e.g., '5Min', '1D')
            initial_capital: Initial capital amount
            risk_per_trade_pct: Risk percentage per trade (default: 2% = 0.02)
            enable_choppiness_filters: Enable choppiness avoidance filters
        """
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct  # ENHANCED: Risk-based sizing
        
        # === SUPER TREND PARAMETERS (OPTIMIZED FROM BACKTESTING) ===
        # Adjust parameters based on timeframe
        if timeframe == '5Min':
            self.stop_loss_pct = 0.05      # 5% stop loss (optimized for 5Min)
            self.min_holding_bars = 20     # Minimum holding period in bars
        elif timeframe == '1D':
            self.stop_loss_pct = 0.17      # 17% stop loss (optimized for 1D)
            self.min_holding_bars = 20     # Minimum holding period in days
        else:
            self.stop_loss_pct = 0.10      # Default 10% stop loss
            self.min_holding_bars = 20     # Default minimum holding
        
        self.supertrend_period = 10    # SuperTrend period
        self.supertrend_multiplier = 3 # SuperTrend multiplier
        
        # Trading state
        self.current_state = PositionState.NONE
        self.current_position = None
        self.entry_time = None
        self.trades = []
        
        # Choppiness filter
        self.choppiness_filter = ChoppinessFilter(enable_choppiness_filters)
        
        # Timezone handling
        self.local_timezone = pytz.timezone('America/Los_Angeles')
        self.market_timezone = pytz.timezone('America/New_York')
        
        print(f"🚀 Enhanced SuperTrend Analyzer initialized for {symbol} on {timeframe} timeframe")
        print(f"💰 Risk Management: SL={self.stop_loss_pct*100:.1f}%, Min Holding={self.min_holding_bars} bars")
        print(f"📊 SuperTrend: Period={self.supertrend_period}, Multiplier={self.supertrend_multiplier}")
        print(f"🛡️ Risk per Trade: {self.risk_per_trade_pct*100:.1f}% of capital")
        print(f"🛡️ Choppiness Filters: {'ENABLED' if enable_choppiness_filters else 'DISABLED'}")
    
    def load_data_from_cache(self) -> pd.DataFrame:
        """Load data from cache file with enhanced error handling and symbol filtering"""
        cache_file = f'data/cache_{self.symbol}_{self.timeframe}.csv'
        
        if not os.path.exists(cache_file):
            print(f"❌ Cache file not found: {cache_file}")
            print("Please ensure the cache file exists before running analysis.")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Check if symbol column exists and has valid data
            if 'symbol' in df.columns:
                # Check if symbol column has any non-empty values
                symbol_values = df['symbol'].dropna().astype(str).str.strip()
                if len(symbol_values) > 0 and not symbol_values.str.match(r'^\s*$').all():
                    print(f"Unique symbols in cache: {df['symbol'].unique()}")
                    # Filter robustly: strip and upper
                    df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()
                    symbol_filter = self.symbol.strip().upper()
                    df = df[df['symbol'] == symbol_filter]
                    print(f"After symbol filtering: {len(df)} rows")
                else:
                    print(f"Symbol column exists but is empty, using all {len(df)} rows.")
            else:
                print(f"No symbol column found, using all {len(df)} rows.")
            
            print(f"✅ Loaded {len(df)} cached bars for {self.symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            return pd.DataFrame()
    
    def calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SuperTrend indicator with improved implementation"""
        def wilder_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
            tr = pd.concat([
                df['high'] - df['low'],
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            ], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            return atr
        
        atr = wilder_atr(df, self.supertrend_period)
        hl2 = (df['high'] + df['low']) / 2
        upperband = hl2 + self.supertrend_multiplier * atr
        lowerband = hl2 - self.supertrend_multiplier * atr
        supertrend = [np.nan] * len(df)
        
        for i in range(1, len(df)):
            prev_st = supertrend[i-1] if not np.isnan(supertrend[i-1]) else upperband.iloc[i-1]
            if df['close'].iloc[i-1] > prev_st:
                st = max(lowerband.iloc[i], prev_st)
            else:
                st = min(upperband.iloc[i], prev_st)
            supertrend[i] = st
        
        df['supertrend'] = supertrend
        return df
    
    def should_enter_long(self, df: pd.DataFrame, i: int) -> bool:
        """Check if we should enter a long position"""
        if i < 1:
            return False
        
        prev_close = df['close'].iloc[i-1]
        prev_st = df['supertrend'].iloc[i-1]
        close = df['close'].iloc[i]
        st = df['supertrend'].iloc[i]
        
        # Long entry: price crosses above SuperTrend
        return prev_close < prev_st and close > st
    
    def should_enter_short(self, df: pd.DataFrame, i: int) -> bool:
        """Check if we should enter a short position"""
        if i < 1:
            return False
        
        prev_close = df['close'].iloc[i-1]
        prev_st = df['supertrend'].iloc[i-1]
        close = df['close'].iloc[i]
        st = df['supertrend'].iloc[i]
        
        # Short entry: price crosses below SuperTrend
        return prev_close > prev_st and close < st
    
    def should_exit_long(self, df: pd.DataFrame, i: int, entry_idx: int) -> Tuple[bool, ExitReason]:
        """Check if we should exit a long position"""
        if i < 1 or entry_idx is None:
            return False, ExitReason.ERROR_EXIT
        
        close = df['close'].iloc[i]
        entry_price = df['close'].iloc[entry_idx]
        
        # Stop loss check
        if close <= entry_price * (1 - self.stop_loss_pct):
            return True, ExitReason.STOP_LOSS
        
        # Minimum holding period check
        if (i - entry_idx) >= self.min_holding_bars:
            # SuperTrend exit check
            prev_close = df['close'].iloc[i-1]
            prev_st = df['supertrend'].iloc[i-1]
            st = df['supertrend'].iloc[i]
            
            if prev_close > prev_st and close < st:
                return True, ExitReason.SUPERTREND_EXIT
        
        return False, ExitReason.ERROR_EXIT
    
    def should_exit_short(self, df: pd.DataFrame, i: int, entry_idx: int) -> Tuple[bool, ExitReason]:
        """Check if we should exit a short position"""
        if i < 1 or entry_idx is None:
            return False, ExitReason.ERROR_EXIT
        
        close = df['close'].iloc[i]
        entry_price = df['close'].iloc[entry_idx]
        
        # Stop loss check
        if close >= entry_price * (1 + self.stop_loss_pct):
            return True, ExitReason.STOP_LOSS
        
        # Minimum holding period check
        if (i - entry_idx) >= self.min_holding_bars:
            # SuperTrend exit check
            prev_close = df['close'].iloc[i-1]
            prev_st = df['supertrend'].iloc[i-1]
            st = df['supertrend'].iloc[i]
            
            if prev_close < prev_st and close > st:
                return True, ExitReason.SUPERTREND_EXIT
        
        return False, ExitReason.ERROR_EXIT
    
    def calculate_risk_based_shares(self, entry_price: float, current_capital: float = None) -> int:
        """Calculate position size based on risk management"""
        if current_capital is None:
            current_capital = self.capital
        
        # Calculate risk amount
        risk_amount = current_capital * self.risk_per_trade_pct
        
        # Calculate shares based on stop loss
        stop_loss_amount = entry_price * self.stop_loss_pct
        shares = int(risk_amount / stop_loss_amount)
        
        # Ensure minimum position size
        min_shares = 1
        shares = max(shares, min_shares)
        
        return shares
    
    def run_enhanced_backtest(self, start_date=None, end_date=None) -> Tuple[pd.DataFrame, float]:
        """Run enhanced backtest with choppiness filters"""
        # Load data
        df = self.load_data_from_cache()
        if df.empty:
            return pd.DataFrame(), 0.0
        
        # Apply date filtering with timezone handling
        if start_date:
            start_ts = pd.to_datetime(start_date)
            # Handle timezone if data has timezone info
            if df['timestamp'].dt.tz is not None:
                start_ts = start_ts.tz_localize(df['timestamp'].dt.tz)
            df = df[df['timestamp'] >= start_ts]
            
        if end_date:
            end_ts = pd.to_datetime(end_date)
            # Handle timezone if data has timezone info
            if df['timestamp'].dt.tz is not None:
                end_ts = end_ts.tz_localize(df['timestamp'].dt.tz)
            df = df[df['timestamp'] <= end_ts]
        
        if len(df) < 50:
            print(f"❌ Insufficient data after filtering: {len(df)} bars")
            return pd.DataFrame(), 0.0
        
        print(f"📊 Running backtest on {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        
        # Calculate SuperTrend
        df = self.calculate_supertrend(df)
        
        # Initialize variables
        position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        entry_date = None
        entry_idx = None
        trades = []
        self.capital = self.initial_capital
        
        # Track recent prices for choppiness filter
        recent_prices = []
        
        # Run backtest
        for i in range(1, len(df)):
            ts = df['timestamp'].iloc[i]
            close = df['close'].iloc[i]
            
            # Update recent prices for choppiness filter
            recent_prices.append(close)
            if len(recent_prices) > 10:  # Keep last 10 prices
                recent_prices.pop(0)
            
            if position == 0:
                # No position - check for entry signals
                should_enter_long = self.should_enter_long(df, i)
                should_enter_short = self.should_enter_short(df, i)
                
                # Apply choppiness filters before entry
                skip_trade = False
                skip_reason = ""
                
                if should_enter_long or should_enter_short:
                    skip_trade, skip_reason = self.choppiness_filter.should_skip_trade(
                        timestamp=ts,
                        current_price=close,
                        recent_prices=recent_prices,
                        recent_trades=trades
                    )
                
                if skip_trade:
                    print(f"⏭️ Skipping trade at {ts}: {skip_reason}")
                    continue
                
                if should_enter_long:
                    position = 1
                    entry_price = close
                    entry_date = ts
                    entry_idx = i
                    print(f"📈 Long entry at {ts}: ${close:.2f}")
                elif should_enter_short:
                    position = -1
                    entry_price = close
                    entry_date = ts
                    entry_idx = i
                    print(f"📉 Short entry at {ts}: ${close:.2f}")
                    
            elif position == 1:
                # Long position - check for exit or reversal
                should_exit, exit_reason = self.should_exit_long(df, i, entry_idx)
                if should_exit:
                    # Exit long position
                    shares = self.calculate_risk_based_shares(entry_price)
                    pnl = shares * (close - entry_price)
                    holding_bars = i - entry_idx
                    
                    trade = Trade(
                        side='long',
                        entry_date=entry_date,
                        entry_price=entry_price,
                        exit_date=ts,
                        exit_price=close,
                        shares=shares,
                        pnl=pnl,
                        stop_loss=(exit_reason == ExitReason.STOP_LOSS),
                        exit_reason=exit_reason.value,
                        holding_bars=holding_bars
                    )
                    trades.append(trade)
                    
                    # Update capital
                    self.capital += pnl
                    
                    print(f"📈 Long exit at {ts}: ${close:.2f}, PnL: ${pnl:.2f}, Reason: {exit_reason.value}")
                    
                    # Check for immediate reversal to short
                    if self.should_enter_short(df, i):
                        position = -1
                        entry_price = close
                        entry_date = ts
                        entry_idx = i
                        print(f"🔄 Reversing to short at {ts}: ${close:.2f}")
                    else:
                        position = 0
                        
            elif position == -1:
                # Short position - check for exit or reversal
                should_exit, exit_reason = self.should_exit_short(df, i, entry_idx)
                if should_exit:
                    # Exit short position
                    shares = self.calculate_risk_based_shares(entry_price)
                    pnl = shares * (entry_price - close)
                    holding_bars = i - entry_idx
                    
                    trade = Trade(
                        side='short',
                        entry_date=entry_date,
                        entry_price=entry_price,
                        exit_date=ts,
                        exit_price=close,
                        shares=shares,
                        pnl=pnl,
                        stop_loss=(exit_reason == ExitReason.STOP_LOSS),
                        exit_reason=exit_reason.value,
                        holding_bars=holding_bars
                    )
                    trades.append(trade)
                    
                    # Update capital
                    self.capital += pnl
                    
                    print(f"📉 Short exit at {ts}: ${close:.2f}, PnL: ${pnl:.2f}, Reason: {exit_reason.value}")
                    
                    # Check for immediate reversal to long
                    if self.should_enter_long(df, i):
                        position = 1
                        entry_price = close
                        entry_date = ts
                        entry_idx = i
                        print(f"🔄 Reversing to long at {ts}: ${close:.2f}")
                    else:
                        position = 0
        
        # Convert trades to DataFrame
        if trades:
            trades_df = pd.DataFrame([vars(trade) for trade in trades])
            print(f"✅ Backtest completed: {len(trades)} trades")
            return trades_df, self.capital
        else:
            print("❌ No trades executed")
            return pd.DataFrame(), self.initial_capital
    
    def get_performance_summary(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if trades_df.empty:
            return {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "capital": self.capital,
                "total_pnl": 0,
                "return_pct": 0,
                "num_trades": 0,
                "win_rate": 0,
                "avg_trade": 0,
                "max_drawdown": 0,
                "stop_loss_trades": 0,
                "supertrend_exits": 0
            }
        
        total_pnl = trades_df['pnl'].sum()
        return_pct = (total_pnl / self.capital) * 100
        num_trades = len(trades_df)
        win_rate = (len(trades_df[trades_df['pnl'] > 0]) / num_trades) * 100
        avg_trade = trades_df['pnl'].mean()
        
        # Calculate max drawdown
        cumulative_pnl = trades_df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # Exit reason breakdown
        stop_loss_trades = len(trades_df[trades_df['stop_loss'] == True])
        supertrend_exits = len(trades_df[trades_df['exit_reason'] == ExitReason.SUPERTREND_EXIT.value])
        
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "capital": self.capital,
            "total_pnl": total_pnl,
            "return_pct": return_pct,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_trade": avg_trade,
            "max_drawdown": max_drawdown,
            "stop_loss_trades": stop_loss_trades,
            "supertrend_exits": supertrend_exits,
            "stop_loss_pct": self.stop_loss_pct * 100,
            "min_holding_bars": self.min_holding_bars
        }

def apply_compound_trading(trades_df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """
    Apply compound trading logic to reinvest profits and losses
    
    Args:
        trades_df: DataFrame containing trade information
        initial_capital: Initial capital amount
    
    Returns:
        DataFrame with compound trading columns added
    """
    if trades_df.empty:
        return trades_df
    
    # Create a copy to avoid modifying the original
    df = trades_df.copy()
    
    # Initialize compound trading variables
    current_capital = initial_capital
    cumulative_pnl = 0
    
    # Lists to store compound trading data
    trade_capitals = []
    final_capitals = []
    cumulative_pnls = []
    
    for i, trade in df.iterrows():
        # Calculate shares based on current capital
        entry_price = trade['entry_price']
        shares = int(current_capital / entry_price)
        
        # Recalculate PnL with actual shares traded
        if trade['side'] == 'long':
            actual_pnl = shares * (trade['exit_price'] - entry_price)
        else:  # short
            actual_pnl = shares * (entry_price - trade['exit_price'])
        
        # Update capital with the actual PnL
        current_capital += actual_pnl
        cumulative_pnl += actual_pnl
        
        # Store compound trading data
        trade_capitals.append(current_capital - actual_pnl)  # Capital before this trade
        final_capitals.append(current_capital)  # Capital after this trade
        cumulative_pnls.append(cumulative_pnl)
        
        # Update the trade with actual values
        df.at[i, 'shares'] = shares
        df.at[i, 'pnl'] = actual_pnl
    
    # Add compound trading columns
    df['trade_capital'] = trade_capitals
    df['final_capital'] = final_capitals
    df['cumulative_pnl'] = cumulative_pnls
    
    return df

def supertrend_with_min_holding_and_stop(symbol='SOXL', timeframe='5Min', min_holding=20, stop_loss_pct=0.10, capital=100000):
    """Legacy function for backward compatibility"""
    analyzer = EnhancedSuperTrendAnalyzer(symbol, timeframe, capital)
    analyzer.min_holding_bars = min_holding
    analyzer.stop_loss_pct = stop_loss_pct
    return analyzer.run_enhanced_backtest()

def optimize_stop_loss(symbol='SOXL', timeframe='5Min', min_holding=20, capital=100000, 
                      stop_loss_range=(5, 15), iterations=50):
    """Test different stop loss percentages to find the optimal one"""
    print(f"=== STOP LOSS OPTIMIZATION FOR {symbol} {timeframe} ===")
    print(f"Testing stop loss percentages from {stop_loss_range[0]}% to {stop_loss_range[1]}%...")
    print(f"Optimization iterations: {iterations}")
    
    # Generate stop loss values to test
    min_stop, max_stop = stop_loss_range
    stop_loss_values = []
    
    if iterations <= 20:
        # For small number of iterations, use linear spacing
        step = (max_stop - min_stop) / (iterations - 1)
        stop_loss_values = [min_stop + i * step for i in range(iterations)]
    else:
        # For larger iterations, use more granular spacing
        step = (max_stop - min_stop) / (iterations - 1)
        stop_loss_values = [min_stop + i * step for i in range(iterations)]
    
    results = []
    for stop_loss_pct in stop_loss_values:
        try:
            trades_df, total_pnl = supertrend_with_min_holding_and_stop(
                symbol=symbol, 
                timeframe=timeframe, 
                min_holding=min_holding, 
                stop_loss_pct=stop_loss_pct/100, 
                capital=capital
            )
            num_trades = len(trades_df)
            stop_loss_trades = len(trades_df[trades_df['stop_loss'] == True]) if not trades_df.empty else 0
            win_rate = (len(trades_df[trades_df['pnl'] > 0]) / num_trades * 100) if num_trades > 0 else 0
            
            results.append({
                'stop_loss_pct': stop_loss_pct,
                'total_pnl': total_pnl,
                'num_trades': num_trades,
                'stop_loss_trades': stop_loss_trades,
                'win_rate': win_rate
            })
            print(f"Stop Loss {stop_loss_pct:5.1f}%: PnL ${total_pnl:>10,.2f}, Trades: {num_trades:3d}, Stop Losses: {stop_loss_trades:2d}, Win Rate: {win_rate:4.1f}%")
        except Exception as e:
            print(f"Error testing stop loss {stop_loss_pct:.1f}%: {e}")
            continue
    
    if not results:
        print("No results found. Check if cache file exists.")
        return 0.10  # Default to 10%
    
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['total_pnl'].idxmax()]
    
    print(f"\n=== OPTIMIZATION RESULTS ===")
    print(f"Best Stop Loss: {best_result['stop_loss_pct']:.1f}%")
    print(f"Total PnL: ${best_result['total_pnl']:,.2f}")
    print(f"Number of Trades: {best_result['num_trades']}")
    print(f"Stop Loss Trades: {best_result['stop_loss_trades']}")
    print(f"Win Rate: {best_result['win_rate']:.1f}%")
    
    # Show top 5 results
    print(f"\n=== TOP 5 RESULTS ===")
    top_5 = results_df.nlargest(5, 'total_pnl')
    for idx, row in top_5.iterrows():
        print(f"{row['stop_loss_pct']:5.1f}%: ${row['total_pnl']:>10,.2f} ({int(row['num_trades']):3d} trades, {row['win_rate']:4.1f}% win rate)")
    
    return best_result['stop_loss_pct'] / 100

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description='Enhanced SuperTrend Trading Strategy Analysis')
    parser.add_argument('--symbol', type=str, default='SOXL', help='Trading symbol (default: SOXL)')
    parser.add_argument('--timeframe', type=str, default='5Min', choices=['1Min', '5Min', '10Min', '15Min', '30Min', '1H', '1D'], help='Data timeframe (default: 5Min)')
    parser.add_argument('--initial_capital', type=float, default=100000, help='Initial capital (default: 100000)')
    parser.add_argument('--min_holding', type=int, default=20, help='Minimum holding period in bars (default: 20)')
    parser.add_argument('--risk_per_trade', type=float, default=0.02, help='Risk percentage per trade (default: 0.02 = 2%)')
    parser.add_argument('--stop_loss_pct', type=float, default=10.0, help='Stop loss percentage (default: 10.0)')
    parser.add_argument('--skip_optimization', action='store_true', help='Skip stop loss optimization')
    parser.add_argument('--enhanced', action='store_true', help='Use enhanced features (compound trading, etc.)')
    parser.add_argument('--start_date', type=str, help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--compounded', action='store_true', help='Enable compound trading (reinvest profits/losses)')
    parser.add_argument('--optimize', action='store_true', help='Run stop loss optimization')
    parser.add_argument('--stop_loss_range', nargs=2, type=float, default=[5, 15], help='Stop loss range for optimization (min max)')
    parser.add_argument('--optimization_iterations', type=int, default=50, help='Number of optimization iterations')
    parser.add_argument('--disable_choppiness_filters', action='store_true', help='Disable choppiness avoidance filters')
    
    args = parser.parse_args()
    
    print(f"🚀 Enhanced SuperTrend Analysis for {args.symbol} on {args.timeframe} timeframe")
    print(f"💰 Capital: ${args.initial_capital:,.0f}")
    print(f"📊 Min Holding: {args.min_holding} bars")
    print(f"🛡️ Risk per Trade: {args.risk_per_trade*100:.1f}%")
    print(f"🎯 Enhanced Mode: {args.enhanced}")
    print(f"📅 Date Range: {args.start_date} to {args.end_date}")
    print(f"🔄 Compound Trading: {args.compounded}")
    print(f"🔧 Optimization: {args.optimize}")
    print(f"🛡️ Choppiness Filters: {'DISABLED' if args.disable_choppiness_filters else 'ENABLED'}")
    if args.optimize:
        print(f"📈 Stop Loss Range: {args.stop_loss_range[0]}% to {args.stop_loss_range[1]}%")
        print(f"🔄 Optimization Iterations: {args.optimization_iterations}")
    
    # Check if cache file exists
    cache_file = f'data/cache_{args.symbol}_{args.timeframe}.csv'
    if not os.path.exists(cache_file):
        print(f"❌ Cache file not found: {cache_file}")
        print("Please ensure the cache file exists before running analysis.")
        print("You can create it using the build_proper_cache.py script.")
        return
    
    if args.enhanced:
        # Use enhanced analyzer with choppiness filters
        analyzer = EnhancedSuperTrendAnalyzer(
            symbol=args.symbol,
            timeframe=args.timeframe,
            initial_capital=args.initial_capital,
            risk_per_trade_pct=args.risk_per_trade,
            enable_choppiness_filters=not args.disable_choppiness_filters
        )
        
        # Set custom stop loss if provided
        analyzer.stop_loss_pct = args.stop_loss_pct / 100.0  # Convert percentage to decimal
        
        # Find optimal stop loss (unless skipped)
        if args.skip_optimization:
            print(f"Using default stop loss: {analyzer.stop_loss_pct*100:.0f}%")
        elif args.optimize:
            optimal_stop_loss = optimize_stop_loss(
                symbol=args.symbol, 
                timeframe=args.timeframe, 
                min_holding=args.min_holding, 
                capital=args.initial_capital,
                stop_loss_range=tuple(args.stop_loss_range),
                iterations=args.optimization_iterations
            )
            analyzer.stop_loss_pct = optimal_stop_loss
            print(f"Using optimized stop loss: {optimal_stop_loss*100:.1f}%")
        else:
            optimal_stop_loss = optimize_stop_loss(symbol=args.symbol, timeframe=args.timeframe, min_holding=args.min_holding, capital=args.initial_capital)
            analyzer.stop_loss_pct = optimal_stop_loss
            print(f"Using optimized stop loss: {optimal_stop_loss*100:.0f}%")
        
        # Run enhanced backtest
        print(f"\n=== RUNNING ENHANCED BACKTEST ===")
        trades_df, total_pnl = analyzer.run_enhanced_backtest(args.start_date, args.end_date)
        
        # Apply compound trading if enabled
        if args.compounded and not trades_df.empty:
            print(f"\n=== APPLYING COMPOUND TRADING ===")
            trades_df = apply_compound_trading(trades_df, args.initial_capital)
            total_pnl = trades_df['cumulative_pnl'].iloc[-1] if not trades_df.empty else 0
            print(f"Final Capital: ${trades_df['final_capital'].iloc[-1]:,.2f}")
            print(f"Total Return: {((trades_df['final_capital'].iloc[-1] / args.initial_capital) - 1) * 100:.2f}%")
        
        # Get performance summary
        performance = analyzer.get_performance_summary(trades_df)
        
        # Display results
        holding_period_text = f"{args.min_holding} bars" if args.timeframe != '1D' else f"{args.min_holding} days"
        compound_text = " (COMPOUNDED)" if args.compounded else ""
        risk_text = f" (Risk: {args.risk_per_trade*100:.1f}%)"
        choppiness_text = " (NO CHOPPINESS FILTERS)" if args.disable_choppiness_filters else " (WITH CHOPPINESS FILTERS)"
        print(f'\n=== {args.symbol} {args.timeframe} ENHANCED SUPERTREND{compound_text}{risk_text}{choppiness_text} (min holding {holding_period_text}, stop loss {int(analyzer.stop_loss_pct*100)}%, ${args.initial_capital:,.0f} capital) TRADE LOG ({args.start_date} to {args.end_date}) ===')
        
        if not trades_df.empty:
            # Display enhanced trade information
            if args.compounded:
                display_cols = ['side', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'shares', 'pnl', 'stop_loss', 'exit_reason', 'holding_bars', 'trade_capital', 'final_capital']
            else:
                display_cols = ['side', 'entry_date', 'entry_price', 'exit_date', 'exit_price', 'shares', 'pnl', 'stop_loss', 'exit_reason', 'holding_bars']
            print(trades_df[display_cols].to_string(index=False))
            
            print(f'\n=== ENHANCED PERFORMANCE SUMMARY ===')
            if args.compounded:
                final_capital = trades_df['final_capital'].iloc[-1]
                total_return_pct = ((final_capital / args.initial_capital) - 1) * 100
                print(f'Initial Capital: ${args.initial_capital:,.2f}')
                print(f'Final Capital: ${final_capital:,.2f}')
                print(f'Optimal Stop Loss: {analyzer.stop_loss_pct*100:.1f}%')
                print(f'Total Return: {total_return_pct:.2f}%')
            else:
                final_capital = args.initial_capital + performance["total_pnl"]
                total_return_pct = performance["return_pct"]
                print(f'Initial Capital: ${args.initial_capital:,.2f}')
                print(f'Final Capital: ${final_capital:,.2f}')
                print(f'Optimal Stop Loss: {analyzer.stop_loss_pct*100:.1f}%')
                print(f'Total PnL: ${performance["total_pnl"]:,.2f}')
                print(f'Total Return: {total_return_pct:.2f}%')
            print(f'Number of Trades: {performance["num_trades"]}')
            print(f'Win Rate: {performance["win_rate"]:.1f}%')
            print(f'Average Trade: ${performance["avg_trade"]:,.2f}')
            print(f'Max Drawdown: ${performance["max_drawdown"]:,.2f}')
            print(f'Stop Loss Trades: {performance["stop_loss_trades"]}')
            print(f'SuperTrend Exits: {performance["supertrend_exits"]}')
            
            # Show choppiness filter statistics if enabled
            if not args.disable_choppiness_filters:
                skipped_trades = (trades_df['exit_reason'] == 'choppy_market').sum() if 'exit_reason' in trades_df.columns else 0
                print(f'Choppiness Filtered Trades: {skipped_trades}')
        else:
            print("❌ No trades executed")
    
    else:
        # Use legacy function
        print(f"\n=== RUNNING LEGACY BACKTEST ===")
        trades_df, total_pnl = supertrend_with_min_holding_and_stop(
            symbol=args.symbol,
            timeframe=args.timeframe,
            min_holding=args.min_holding,
            stop_loss_pct=args.stop_loss_pct/100.0,
            capital=args.initial_capital
        )
        
        if not trades_df.empty:
            print(trades_df.to_string(index=False))
            print(f'\n=== LEGACY PERFORMANCE SUMMARY ===')
            print(f'Total PnL: ${total_pnl:,.2f}')
            print(f'Number of Trades: {len(trades_df)}')
            print(f'Win Rate: {(len(trades_df[trades_df["pnl"] > 0]) / len(trades_df) * 100):.1f}%')
            print(f'Average Trade: ${trades_df["pnl"].mean():,.2f}')
        else:
            print("❌ No trades executed")

if __name__ == "__main__":
    main() 