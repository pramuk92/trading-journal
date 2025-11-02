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
        """Parse Plus500 PDF statement and extract trades"""
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        return self._extract_trades_simple(full_text)
    
    def _extract_trades_simple(self, text: str) -> pd.DataFrame:
        """Simple but robust trade extraction"""
        trades = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for date patterns (MM/DD/YYYY)
            date_match = re.match(r'^(\d{2}/\d{2}/\d{4})$', line)
            if date_match:
                trade_date_str = date_match.group(1)
                
                # Look ahead in next lines for trade information
                trade_info = self._find_trade_info(trade_date_str, lines[i:min(i+10, len(lines))])
                if trade_info:
                    trades.append(trade_info)
                    # Skip ahead since we found a trade
                    i += 3
                else:
                    i += 1
            else:
                i += 1
        
        return pd.DataFrame(trades)
    
    def _find_trade_info(self, trade_date: str, context_lines: List[str]) -> Optional[Dict]:
        """Find trade information in context lines following a date"""
        context_text = ' | '.join([line.strip() for line in context_lines if line.strip()])
        
        # Look for PNL entries
        pnl_matches = re.finditer(r'PNL\s+USD\s+([\(\)\d\.,]+)\*?', context_text)
        
        for pnl_match in pnl_matches:
            pnl_str = pnl_match.group(1)
            pnl = self._parse_amount(pnl_str)
            
            # Extract instrument from the context
            instrument = self._extract_instrument(context_text)
            
            # Extract exchange
            exchange = "UNKNOWN"
            if 'CBOT' in context_text:
                exchange = 'CBOT'
            elif 'CME' in context_text:
                exchange = 'CME'
            
            # Find corresponding commission
            commission = self._find_commission(context_text)
            
            if instrument:
                return {
                    'trade_date': datetime.strptime(trade_date, '%m/%d/%Y'),
                    'instrument': instrument,
                    'exchange': exchange,
                    'pnl': pnl,
                    'commission': commission,
                    'net_pnl': pnl + commission,
                    'direction': 'LONG' if pnl > 0 else 'SHORT'
                }
        
        return None
    
    def _extract_instrument(self, text: str) -> str:
        """Extract instrument name from text"""
        # Common Plus500 futures instruments
        patterns = [
            r'Micro E-mini Dow Jones Industrial Average Index Futures',
            r'Micro Nikkei \(USD\) Futures',
            r'Dec \d+ Micro E-mini Dow Jones Industrial Average Index Futures',
            r'Dec \d+ Micro Nikkei \(USD\) Futures',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return "Unknown Instrument"
    
    def _find_commission(self, text: str) -> float:
        """Find commission amount in text"""
        comm_match = re.search(r'FEE/COMM\s+USD\s+([\(\)\d\.,]+)\*?', text)
        if comm_match:
            return self._parse_amount(comm_match.group(1))
        return 0.0
    
    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string with bracket notation for losses"""
        amount_str = str(amount_str).replace('*', '').strip().replace(',', '')
        
        # Handle bracket notation for negatives
        if amount_str.startswith('(') and amount_str.endswith(')'):
            number_str = amount_str[1:-1]
            return -float(number_str)
        else:
            return float(amount_str)
    
    def debug_pdf_structure(self, pdf_path: str) -> str:
        """Debug method to see PDF structure"""
        doc = fitz.open(pdf_path)
        debug_info = "PDF STRUCTURE ANALYSIS:\n\n"
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            lines = text.split('\n')
            
            debug_info += f"=== PAGE {page_num + 1} ===\n"
            debug_info += f"Total lines: {len(lines)}\n"
            
            # Find activity section
            for i, line in enumerate(lines):
                if "YOUR ACTIVITY THIS MONTH" in line:
                    debug_info += f"Found ACTIVITY section at line {i}\n"
                    # Show context around activity section
                    start = max(0, i-2)
                    end = min(len(lines), i+10)
                    debug_info += "Context lines:\n"
                    for j in range(start, end):
                        debug_info += f"  {j}: {lines[j]}\n"
                    break
            
            # Find dates and potential trades
            date_lines = []
            for i, line in enumerate(lines):
                if re.match(r'\d{2}/\d{2}/\d{4}', line):
                    date_lines.append(f"  Line {i}: {line}")
            
            if date_lines:
                debug_info += f"Date lines found: {len(date_lines)}\n"
                debug_info += "\n".join(date_lines[:10]) + "\n"
        
        doc.close()
        return debug_info

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
            st.session_state.trades_df = parser.parse_pdf("temp_statement.pdf")
            
            if not st.session_state.trades_df.empty:
                st.sidebar.success(f"✅ Successfully parsed {len(st.session_state.trades_df)} trades!")
                st.sidebar.dataframe(st.session_state.trades_df[['trade_date', 'instrument', 'net_pnl']].head())
            else:
                st.sidebar.warning("⚠️ No trades found with current parser.")
                
                # Enhanced debugging
                if st.sidebar.button("Debug PDF Structure"):
                    debug_info = parser.debug_pdf_structure("temp_statement.pdf")
                    st.sidebar.text_area("PDF Structure Analysis:", debug_info, height=400)
                
                # Also show raw text for manual inspection
                doc = fitz.open("temp_statement.pdf")
                raw_text = ""
                for page in doc:
                    raw_text += page.get_text()
                doc.close()
                
                st.sidebar.text_area("First 1500 chars of raw PDF text:", raw_text[:1500], height=300)
            
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
