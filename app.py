import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import fitz  # PyMuPDF
import base64
from typing import List, Dict, Optional

# Page configuration
st.set_page_config(
    page_title="Plus500 Trading Journal",
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
    .metric-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00D4AA;
    }
</style>
""", unsafe_allow_html=True)

class Plus500Parser:
    def parse_pdf(self, pdf_path: str) -> pd.DataFrame:
        """Parse Plus500 PDF statement and extract trades from activity table"""
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        return self._extract_trades_from_activity(full_text)
    
    def _extract_trades_from_activity(self, text: str) -> pd.DataFrame:
        """Extract trades from YOUR ACTIVITY THIS MONTH section"""
        trades = []
        
        # Split by lines and find the ACTIVITY section
        lines = text.split('\n')
        in_activity_section = False
        activity_started = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Find the start of activity section
            if "YOUR ACTIVITY THIS MONTH" in line:
                in_activity_section = True
                continue
            
            # Look for table header to know when data starts
            if in_activity_section and "TRADE DATE" in line and "BUY" in line and "DESCRIPTION" in line:
                activity_started = True
                continue
            
            # Process data rows in activity section
            if in_activity_section and activity_started:
                # Stop when we hit the next major section
                if line and ("YOUR CASH ACTIVITY" in line or "ACCOUNT SUMMARY" in line):
                    break
                
                # Process trade rows - they contain dates and PNL/FEE info
                if self._is_trade_row(line, lines, i):
                    trade = self._parse_trade_row(line, lines, i)
                    if trade:
                        trades.append(trade)
        
        return pd.DataFrame(trades)
    
    def _is_trade_row(self, line: str, lines: List[str], current_index: int) -> bool:
        """Check if this line is a trade row (contains date and PNL/FEE)"""
        # Check if line has a date pattern
        date_match = re.match(r'(\d{2}/\d{2}/\d{4})', line)
        if not date_match:
            return False
        
        # Check if this line or nearby lines contain PNL or FEE/COMM
        context = ' '.join(lines[max(0, current_index):min(len(lines), current_index + 3)])
        return 'PNL' in context or 'FEE/COMM' in context
    
    def _parse_trade_row(self, line: str, lines: List[str], current_index: int) -> Optional[Dict]:
        """Parse a trade row from the activity table"""
        try:
            # Extract date
            date_match = re.search(r'(\d{2}/\d{2}/\d{4})', line)
            if not date_match:
                return None
            
            trade_date = datetime.strptime(date_match.group(1), '%m/%d/%Y')
            
            # Look for PNL or FEE/COMM in current and next lines
            pnl = 0.0
            commission = 0.0
            instrument = "Unknown"
            exchange = "UNKNOWN"
            
            # Check current line and next 2 lines for trade info
            for j in range(current_index, min(current_index + 3, len(lines))):
                context_line = lines[j]
                
                # Extract PNL
                pnl_match = re.search(r'PNL\s+USD\s+([\(\)\d\.,]+)\*?', context_line)
                if pnl_match:
                    pnl_str = pnl_match.group(1)
                    pnl = self._parse_amount(pnl_str)
                    
                    # Extract instrument from context
                    instrument = self._extract_instrument_from_context(context_line)
                    
                    # Extract exchange
                    if 'CBOT' in context_line:
                        exchange = 'CBOT'
                    elif 'CME' in context_line:
                        exchange = 'CME'
                
                # Extract Commission
                comm_match = re.search(r'FEE/COMM\s+USD\s+([\(\)\d\.,]+)\*?', context_line)
                if comm_match:
                    commission_str = comm_match.group(1)
                    commission = self._parse_amount(commission_str)
            
            # Only return trade if we found meaningful data
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
            st.sidebar.warning(f"Error parsing row: {line} - {str(e)}")
            return None
    
    def _extract_instrument_from_context(self, text: str) -> str:
        """Extract instrument name from context"""
        instruments = [
            'Micro E-mini Dow Jones Industrial Average Index Futures',
            'Micro Nikkei (USD) Futures',
            'Micro E-mini S&P 500 Futures',
            'Micro Gold Futures',
            'Micro Silver Futures',
            'Micro Crude Oil Futures'
        ]
        
        for instrument in instruments:
            if instrument in text:
                return instrument
        
        # Fallback: extract anything that looks like an instrument name
        match = re.search(r'(Dec\s+\d+\s+.*?Futures|.*?Futures)', text)
        return match.group(1) if match else "Unknown Instrument"
    
    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string with bracket notation for losses"""
        amount_str = str(amount_str).replace('*', '').strip().replace(',', '')
        
        # Handle bracket notation for negatives
        if amount_str.startswith('(') and amount_str.endswith(')'):
            number_str = amount_str[1:-1]
            return -float(number_str)
        else:
            return float(amount_str)

class TradingAnalysis:
    def __init__(self, trades_df: pd.DataFrame):
        self.trades_df = trades_df.copy()
        if not self.trades_df.empty:
            self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess trade data for analysis"""
        self.trades_df['win'] = self.trades_df['net_pnl'] > 0
        self.trades_df['cumulative_pnl'] = self.trades_df['net_pnl'].cumsum()
        self.trades_df['trade_day'] = self.trades_df['trade_date'].dt.date
    
    def get_summary_metrics(self) -> Dict:
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
        
        # Profit factor: gross profits / gross losses
        gross_profit = self.trades_df[self.trades_df['net_pnl'] > 0]['net_pnl'].sum()
        gross_loss = abs(self.trades_df[self.trades_df['net_pnl'] < 0]['net_pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
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
            'avg_trade': avg_trade,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }

def create_pnl_chart(trades_df: pd.DataFrame):
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

def create_win_loss_pie(trades_df: pd.DataFrame):
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

def create_instrument_performance(trades_df: pd.DataFrame):
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

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Plus500 Futures Trading Journal</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Data Input")
    st.sidebar.markdown("Upload your Plus500 monthly statement PDF")
    
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
            st.session_state.trades_df = parser.parse_pdf("temp_statement.pdf")
            
            if not st.session_state.trades_df.empty:
                st.sidebar.success(f"✅ Successfully parsed {len(st.session_state.trades_df)} trades!")
                st.sidebar.dataframe(st.session_state.trades_df[['trade_date', 'instrument', 'net_pnl']].head())
            else:
                st.sidebar.warning("⚠️ No trades found. The parser might need adjustment for your specific PDF format.")
                # Debug: Show what the parser sees
                if st.sidebar.button("Debug PDF Content"):
                    doc = fitz.open("temp_statement.pdf")
                    debug_text = ""
                    for page in doc:
                        debug_text += page.get_text()
                    doc.close()
                    st.sidebar.text_area("First 2000 chars of PDF:", debug_text[:2000], height=300)
            
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
    ## Welcome to Your Plus500 Trading Journal!
    
    This app helps you analyze your futures trading performance from Plus500 statements.
    
    ### How to use:
    1. **Upload your Plus500 monthly statement PDF** using the sidebar
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
    
    # Initialize analysis
    analysis = TradingAnalysis(trades_df)
    
    # Summary metrics
    metrics = analysis.get_summary_metrics()
    
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
    
    # Trade data table
    st.markdown("## 📋 Trade Details")
    
    # Format the dataframe for display
    display_df = trades_df.copy()
    display_df['trade_date'] = display_df['trade_date'].dt.strftime('%Y-%m-%d')
    display_df['pnl'] = display_df['pnl'].round(2)
    display_df['commission'] = display_df['commission'].round(2)
    display_df['net_pnl'] = display_df['net_pnl'].round(2)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Instrument analysis
    st.markdown("## 🔍 Instrument Analysis")
    instrument_analysis = trades_df.groupby('instrument').agg({
        'net_pnl': ['count', 'sum', 'mean', 'std'],
        'win': 'mean'
    }).round(2)
    st.dataframe(instrument_analysis, use_container_width=True)
    
    # Export data
    st.markdown("## 💾 Export Data")
    
    if st.button("Download Trade Data as CSV"):
        csv = trades_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="plus500_trades.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
