import streamlit as st
import pandas as pd
import plotly.express as px
from pdf_parser import Plus500Parser
from utils.analysis import TradingAnalysis
from utils.visualization import TradingVisualizations
import base64
import io

# Page configuration
st.set_page_config(
    page_title="Trading Journal",
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
    .positive { color: #00D4AA; }
    .negative { color: #FF6B6B; }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Plus500 Futures Trading Journal</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Data Input")
    st.sidebar.markdown("Upload your monthly statement PDF")
    
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
            
            st.sidebar.success(f"✅ Successfully parsed {len(st.session_state.trades_df)} trades!")
            
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
    viz = TradingVisualizations()
    
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
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "∞")
    
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
        fig = viz.create_pnl_chart(trades_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = viz.create_win_loss_pie(trades_df)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = viz.create_instrument_performance(trades_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = viz.create_daily_heatmap(trades_df)
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
    instrument_analysis = analysis.get_instrument_analysis()
    if not instrument_analysis.empty:
        st.dataframe(instrument_analysis, use_container_width=True)
    
    # Daily analysis
    st.markdown("## 📅 Daily Performance")
    daily_analysis = analysis.get_daily_analysis()
    if not daily_analysis.empty:
        st.dataframe(daily_analysis, use_container_width=True)
    
    # Export data
    st.markdown("## 💾 Export Data")
    
    if st.button("Download Trade Data as CSV"):
        csv = trades_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="plus500_trades.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
