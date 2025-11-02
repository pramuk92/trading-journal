import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class TradingVisualizations:
    @staticmethod
    def create_pnl_chart(trades_df):
        """Create cumulative P&L chart"""
        if trades_df.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trades_df['trade_date'],
            y=trades_df['cumulative_pnl'],
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(color='#00D4AA', width=3)
        ))
        
        fig.update_layout(
            title='Cumulative P&L Over Time',
            xaxis_title='Trade Date',
            yaxis_title='Cumulative P&L (USD)',
            template='plotly_dark'
        )
        
        return fig
    
    @staticmethod
    def create_win_loss_pie(trades_df):
        """Create win/loss pie chart"""
        if trades_df.empty:
            return go.Figure()
        
        win_count = trades_df['win'].sum()
        loss_count = len(trades_df) - win_count
        
        fig = go.Figure(data=[go.Pie(
            labels=['Winning Trades', 'Losing Trades'],
            values=[win_count, loss_count],
            hole=.3,
            marker_colors=['#00D4AA', '#FF6B6B']
        )])
        
        fig.update_layout(title='Win/Loss Distribution')
        return fig
    
    @staticmethod
    def create_instrument_performance(trades_df):
        """Create bar chart of performance by instrument"""
        if trades_df.empty:
            return go.Figure()
        
        instrument_pnl = trades_df.groupby('instrument')['net_pnl'].sum().sort_values()
        
        fig = go.Figure(data=[go.Bar(
            x=instrument_pnl.values,
            y=instrument_pnl.index,
            orientation='h',
            marker_color=['#00D4AA' if x > 0 else '#FF6B6B' for x in instrument_pnl.values]
        )])
        
        fig.update_layout(
            title='Total P&L by Instrument',
            xaxis_title='Net P&L (USD)',
            yaxis_title='Instrument'
        )
        
        return fig
    
    @staticmethod
    def create_daily_heatmap(trades_df):
        """Create heatmap of trading activity by day of week"""
        if trades_df.empty:
            return go.Figure()
        
        trades_df = trades_df.copy()
        trades_df['day_of_week'] = trades_df['trade_date'].dt.day_name()
        trades_df['week_number'] = trades_df['trade_date'].dt.isocalendar().week
        
        daily_pnl = trades_df.groupby('day_of_week')['net_pnl'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        fig = go.Figure(data=[go.Bar(
            x=daily_pnl.index,
            y=daily_pnl.values,
            marker_color=['#00D4AA' if x > 0 else '#FF6B6B' for x in daily_pnl.values]
        )])
        
        fig.update_layout(
            title='Average P&L by Day of Week',
            xaxis_title='Day of Week',
            yaxis_title='Average P&L (USD)'
        )
        
        return fig
