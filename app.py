import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import fitz  # PyMuPDF
import base64
from typing import List, Dict, Optional, Tuple

# Page configuration
st.set_page_config(
    page_title="Futures Trading Journal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00D4AA;
        text-align: center;
        margin-bottom: 2rem;
    }
    .broker-note {
        background-color: #2E2E2E;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #FFA500;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

class Plus500Parser:
    def parse_pdf(self, pdf_path: str) -> pd.DataFrame:
        """Parse Plus500 PDF statement and extract trades from the line-by-line table"""
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        return self._extract_trades_from_lines(full_text)
    
    def _extract_trades_from_lines(self, text: str) -> pd.DataFrame:
        """Extract trades from the line-by-line table format"""
        trades = []
        lines = text.split('\n')
        
        # Find the start of the activity section
        activity_start = -1
        for i, line in enumerate(lines):
            if "YOUR ACTIVITY THIS MONTH" in line:
                activity_start = i
                break
        
        if activity_start == -1:
            return pd.DataFrame()
        
        # Process lines in the activity section
        i = activity_start + 1
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for date lines (these mark the start of trade entries)
            if re.match(r'^\d{2}/\d{2}/\d{4}$', line):
                trade_data = self._parse_trade_block(lines, i)
                if trade_data:
                    trades.append(trade_data)
                    # Skip ahead since we processed a block
                    i += self._get_trade_block_size(lines, i)
                else:
                    i += 1
            else:
                i += 1
        
        return pd.DataFrame(trades)
    
    def _parse_trade_block(self, lines: List[str], start_idx: int) -> Optional[Dict]:
        """Parse a block of lines that represent one trade entry"""
        try:
            # The first line should be the date
            trade_date_str = lines[start_idx].strip()
            trade_date = datetime.strptime(trade_date_str, '%m/%d/%Y')
            
            # Look ahead to find PNL and commission information
            pnl = 0.0
            commission = 0.0
            instrument = "Unknown"
            exchange = "UNKNOWN"
            
            # Scan next 15 lines for trade information
            for i in range(start_idx + 1, min(start_idx + 15, len(lines))):
                line = lines[i].strip()
                
                # Look for instrument names
                if not instrument or instrument == "Unknown":
                    instrument = self._extract_instrument_from_line(line)
                
                # Look for exchange
                if exchange == "UNKNOWN":
                    if 'CBOT' in line:
                        exchange = 'CBOT'
                    elif 'CME' in line:
                        exchange = 'CME'
                
                # Look for PNL
                if pnl == 0 and 'PNL' in line:
                    # The amount might be in this line or next lines
                    pnl = self._find_amount_nearby(lines, i, 'PNL')
                
                # Look for Commission
                if commission == 0 and ('FEE/COMM' in line or 'COMM' in line):
                    commission = self._find_amount_nearby(lines, i, 'COMM')
            
            # Only return if we found meaningful data
            if pnl != 0 or commission != 0:
                return {
                    'trade_date': trade_date,
                    'instrument': instrument,
                    'exchange': exchange,
                    'pnl': pnl,
                    'commission': commission,
                    'net_pnl': pnl + commission,
                    'direction': 'LONG' if pnl > 0 else 'SHORT'
                }
            
            return None
            
        except Exception as e:
            return None
    
    def _find_amount_nearby(self, lines: List[str], idx: int, pattern: str) -> float:
        """Find amount near a line containing specific pattern"""
        # Check current line and next 3 lines for amount
        for i in range(idx, min(idx + 4, len(lines))):
            line = lines[i].strip()
            
            # Look for amount patterns
            amount_match = re.search(r'USD\s+([\(\)\d\.,]+)\*?', line)
            if amount_match:
                return self._parse_amount(amount_match.group(1))
            
            # Also check for standalone amounts
            amount_match = re.search(r'^([\(\)\d\.,]+)\*?$', line)
            if amount_match:
                return self._parse_amount(amount_match.group(1))
        
        return 0.0
    
    def _extract_instrument_from_line(self, line: str) -> str:
        """Extract instrument name from a line"""
        instruments = [
            'Micro E-mini Dow Jones Industrial Average Index Futures',
            'Micro Nikkei (USD) Futures',
            'Dec 25 Micro E-mini Dow Jones Industrial Average Index Futures',
            'Dec 25 Micro Nikkei (USD) Futures',
        ]
        
        for instrument in instruments:
            if instrument in line:
                return instrument
        
        # Check for partial matches
        if 'Micro E-mini Dow' in line:
            return 'Micro E-mini Dow Jones Industrial Average Index Futures'
        elif 'Micro Nikkei' in line:
            return 'Micro Nikkei (USD) Futures'
        
        return "Unknown Instrument"
    
    def _get_trade_block_size(self, lines: List[str], start_idx: int) -> int:
        """Estimate how many lines a trade block occupies"""
        # Look for the next date or end of section
        for i in range(start_idx + 1, min(start_idx + 20, len(lines))):
            if re.match(r'^\d{2}/\d{2}/\d{4}$', lines[i].strip()):
                return i - start_idx
            if 'YOUR CASH ACTIVITY' in lines[i] or 'ACCOUNT SUMMARY' in lines[i]:
                return i - start_idx
        return 10  # Default block size
    
    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string with bracket notation for losses"""
        amount_str = str(amount_str).replace('*', '').strip().replace(',', '')
        
        # Handle bracket notation for negatives
        if amount_str.startswith('(') and amount_str.endswith(')'):
            number_str = amount_str[1:-1]
            return -float(number_str)
        else:
            return float(amount_str)

def preprocess_trades_data(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess trade data for analysis - add calculated columns"""
    if trades_df.empty:
        return trades_df
    
    df = trades_df.copy()
    df['win'] = df['net_pnl'] > 0
    df['cumulative_pnl'] = df['net_pnl'].cumsum()
    df['trade_day'] = df['trade_date'].dt.date
    df['day_of_week'] = df['trade_date'].dt.day_name()
    df['week_number'] = df['trade_date'].dt.isocalendar().week
    
    return df

def get_summary_metrics(trades_df: pd.DataFrame) -> Dict:
    """Calculate key trading performance metrics"""
    if trades_df.empty:
        return {}
    
    total_trades = len(trades_df)
    winning_trades = trades_df['win'].sum()
    losing_trades = total_trades - winning_trades
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = trades_df['net_pnl'].sum()
    avg_win = trades_df[trades_df['win']]['net_pnl'].mean() if winning_trades > 0 else 0
    avg_loss = trades_df[~trades_df['win']]['net_pnl'].mean() if losing_trades > 0 else 0
    
    # Profit factor: gross profits / gross losses
    gross_profit = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
    gross_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    max_win = trades_df['net_pnl'].max()
    max_loss = trades_df['net_pnl'].min()
    avg_trade = trades_df['net_pnl'].mean()
    
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
        'avg_trade': avg_trade,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
    }

def create_pnl_chart(trades_df: pd.DataFrame):
    """Create cumulative P&L chart"""
    if trades_df.empty:
        # Return empty chart with message
        fig = go.Figure()
        fig.add_annotation(text="No trade data available",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Make sure we have the cumulative_pnl column
    if 'cumulative_pnl' not in trades_df.columns:
        trades_df = preprocess_trades_data(trades_df)
    
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

def create_win_loss_pie(trades_df: pd.DataFrame):
    """Create win/loss pie chart"""
    if trades_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No trade data available",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Make sure we have the win column
    if 'win' not in trades_df.columns:
        trades_df = preprocess_trades_data(trades_df)
    
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

def create_instrument_performance(trades_df: pd.DataFrame):
    """Create bar chart of performance by instrument"""
    if trades_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No trade data available",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
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

def create_daily_performance(trades_df: pd.DataFrame):
    """Create daily performance chart"""
    if trades_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No trade data available",
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Make sure we have the trade_day column
    if 'trade_day' not in trades_df.columns:
        trades_df = preprocess_trades_data(trades_df)
    
    daily_pnl = trades_df.groupby('trade_day')['net_pnl'].sum()
    
    fig = go.Figure(data=[go.Bar(
        x=daily_pnl.index,
        y=daily_pnl.values,
        marker_color=['#00D4AA' if x > 0 else '#FF6B6B' for x in daily_pnl.values]
    )])
    
    fig.update_layout(
        title='Daily P&L',
        xaxis_title='Date',
        yaxis_title='Daily P&L (USD)'
    )
    
    return fig

def main():
    # Header with generic title
    st.markdown('<h1 class="main-header">📈 Futures Trading Journal</h1>', unsafe_allow_html=True)
    
    # Broker compatibility note
    st.markdown("""
    <div class="broker-note">
    <strong>Note:</strong> Currently supports Plus500 US futures statements. More brokers coming soon!
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Data Input")
    st.sidebar.markdown("Upload your futures trading statement PDF")
    
    uploaded_file = st.sidebar.file_uploader("Choose PDF file", type="pdf")
    
    # Initialize session state
    if 'trades_df' not in st.session_state:
        st.session_state.trades_df = pd.DataFrame()
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with open("temp_statement.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Parse PDF
            parser = Plus500Parser()
            raw_trades_df = parser.parse_pdf("temp_statement.pdf")
            
            # Preprocess the data for analysis
            if not raw_trades_df.empty:
                st.session_state.trades_df = preprocess_trades_data(raw_trades_df)
                st.sidebar.success(f"✅ Successfully parsed {len(st.session_state.trades_df)} trades!")
                # Show a preview of the parsed data
                preview_df = st.session_state.trades_df[['trade_date', 'instrument', 'net_pnl']].copy()
                preview_df['trade_date'] = preview_df['trade_date'].dt.strftime('%Y-%m-%d')
                preview_df['net_pnl'] = preview_df['net_pnl'].round(2)
                st.sidebar.dataframe(preview_df.head())
            else:
                st.sidebar.warning("⚠️ No trades found. The PDF format might need additional parsing rules.")
                
        except Exception as e:
            st.sidebar.error(f"Error parsing PDF: {str(e)}")
    
    # Main content
    if not st.session_state.trades_df.empty:
        display_analysis(st.session_state.trades_df)
    else:
        display_welcome()

def display_welcome():
    """Display welcome message and instructions"""
    st.markdown("""
    ## Welcome to Your Futures Trading Journal!
    
    This app helps you analyze your futures trading performance from broker statements.
    
    ### How to use:
    1. **Upload your futures trading statement PDF** using the sidebar
    2. **View automated analysis** of your trading performance
    3. **Explore visualizations** to understand your trading patterns
    
    ### Features:
    - 📊 **Performance Metrics**: Win rate, profit factor, average P&L
    - 📈 **Interactive Charts**: P&L over time, instrument performance
    - 🔍 **Trade Analysis**: Breakdown by instrument, day of week
    - 💰 **Risk Metrics**: Max drawdown, risk-adjusted returns
    
    *Upload a PDF to get started!*
    """)

def display_analysis(trades_df):
    """Display trading analysis and visualizations"""
    
    # Make sure data is preprocessed
    if 'cumulative_pnl' not in trades_df.columns:
        trades_df = preprocess_trades_data(trades_df)
    
    # Summary metrics
    metrics = get_summary_metrics(trades_df)
    
    # Display key metrics
    st.markdown("## 📊 Trading Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total P&L", f"${metrics['total_pnl']:.2f}", 
                 delta_color="normal" if metrics['total_pnl'] >= 0 else "inverse")
    
    with col2:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    
    with col3:
        st.metric("Total Trades", metrics['total_trades'])
    
    with col4:
        pf_display = f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "∞"
        st.metric("Profit Factor", pf_display)
    
    # Detailed metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Win", f"${metrics['avg_win']:.2f}")
    
    with col2:
        st.metric("Average Loss", f"${metrics['avg_loss']:.2f}")
    
    with col3:
        st.metric("Max Win", f"${metrics['max_win']:.2f}")
    
    with col4:
        st.metric("Max Loss", f"${metrics['max_loss']:.2f}")
    
    # Visualizations
    st.markdown("## 📈 Performance Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_pnl_chart(trades_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_win_loss_pie(trades_df)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_instrument_performance(trades_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_daily_performance(trades_df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade data table
    st.markdown("## 📋 Trade Details")
    
    # Format the dataframe for display
    display_df = trades_df.copy()
    display_df['trade_date'] = display_df['trade_date'].dt.strftime('%Y-%m-%d')
    display_df['pnl'] = display_df['pnl'].round(2)
    display_df['commission'] = display_df['commission'].round(2)
    display_df['net_pnl'] = display_df['net_pnl'].round(2)
    
    st.dataframe(display_df[['trade_date', 'instrument', 'exchange', 'pnl', 'commission', 'net_pnl', 'direction']], 
                 use_container_width=True)
    
    # Instrument analysis
    st.markdown("## 🔍 Instrument Analysis")
    instrument_analysis = trades_df.groupby('instrument').agg({
        'net_pnl': ['count', 'sum', 'mean', 'std'],
        'win': 'mean'
    }).round(2)
    if not instrument_analysis.empty:
        st.dataframe(instrument_analysis, use_container_width=True)
    
    # Export data
    st.markdown("## 💾 Export Data")
    
    if st.button("Download Trade Data as CSV"):
        csv = trades_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="futures_trades.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
