import pandas as pd
import numpy as np
from datetime import datetime

class TradingAnalysis:
    def __init__(self, trades_df):
        self.trades_df = trades_df.copy()
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess trade data for analysis"""
        if not self.trades_df.empty:
            self.trades_df['win'] = self.trades_df['net_pnl'] > 0
            self.trades_df['win_amount'] = self.trades_df[self.trades_df['win']]['net_pnl']
            self.trades_df['loss_amount'] = self.trades_df[~self.trades_df['win']]['net_pnl']
            self.trades_df['cumulative_pnl'] = self.trades_df['net_pnl'].cumsum()
    
    def get_summary_metrics(self):
        """Calculate key trading performance metrics"""
        if self.trades_df.empty:
            return {}
        
        total_trades = len(self.trades_df)
        winning_trades = self.trades_df['win'].sum()
        losing_trades = total_trades - winning_trades
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = self.trades_df['net_pnl'].sum()
        avg_win = self.trades_df[self.trades_df['win']]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = self.trades_df[~self.trades_df['win']]['net_pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
        
        max_win = self.trades_df['net_pnl'].max()
        max_loss = self.trades_df['net_pnl'].min()
        avg_trade = self.trades_df['net_pnl'].mean()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_win': max_win,
            'max_loss': max_loss,
            'avg_trade': avg_trade
        }
    
    def get_instrument_analysis(self):
        """Analyze performance by instrument"""
        if self.trades_df.empty:
            return pd.DataFrame()
        
        return self.trades_df.groupby('instrument').agg({
            'net_pnl': ['count', 'sum', 'mean', 'std'],
            'win': 'mean'
        }).round(2)
    
    def get_daily_analysis(self):
        """Analyze performance by day"""
        if self.trades_df.empty:
            return pd.DataFrame()
        
        daily = self.trades_df.groupby(self.trades_df['trade_date'].dt.date).agg({
            'net_pnl': ['sum', 'count'],
            'win': 'mean'
        }).round(2)
        
        daily.columns = ['daily_pnl', 'trades_count', 'win_rate']
        return daily
