# trading_journal_app.py
# Generic Trading Journal Analyzer - Compatible with multiple CSV formats
# Note: Currently pre-configured for broker CSV exports similar to Plus500 format
# This tool is independent and not affiliated with any brokerage

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Trading Journal Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Header
    st.title("📊 Trading Journal Analyzer")
    st.markdown("""
    Upload your trading history CSV file to analyze your performance.
    *Currently compatible with CSV exports from various brokers.*
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Process data
            df, success = process_data(uploaded_file)
            
            if success:
                display_analysis(df)
            else:
                st.error("Please check your CSV format. Required columns: Date, Action, Instrument, NetPL")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Show sample format expectations
        st.info("""
        **Expected CSV Format:**
        - Date, Action, Amount, Instrument, AverageOpenPrice, ClosePrice, GrossPL, NetPL
        - Example: `10/30/2025 8:20 PM,Buy,1,Micro Nikkei Dec 25,52340,52035,($152.50),($154.38)`
        """)

def process_data(uploaded_file):
    """Process and clean the uploaded CSV data"""
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        # Required columns check
        required_cols = ['Date', 'Action', 'Instrument', 'NetPl']
        if not all(col in df.columns for col in required_cols):
            return None, False
        
        # Data cleaning
        df = df.copy()
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Clean P&L columns - convert from currency string to float
        def clean_currency(value):
            if isinstance(value, str):
                value = value.replace('$', '').replace(',', '')
                if '(' in value and ')' in value:
                    return -float(value.strip('()'))
                else:
                    return float(value)
            return float(value)
        
        # Apply cleaning to P&L columns
        pnl_columns = ['NetPl', 'GrossPl'] 
        for col in pnl_columns:
            if col in df.columns:
                df[f'{col}_Clean'] = df[col].apply(clean_currency)
        
        # Extract instrument base name
        df['Instrument_Base'] = df['Instrument'].str.split('(').str[0].str.strip()
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Calculate cumulative P&L
        df['Cumulative_NetPL'] = df['NetPl_Clean'].cumsum()
        
        # Add trade result
        df['Result'] = df['NetPl_Clean'].apply(lambda x: 'Win' if x > 0 else 'Loss' if x < 0 else 'BreakEven')
        
        return df, True
        
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return None, False

def display_analysis(df):
    """Display all analysis components"""
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Instrument filter
    instruments = ['All'] + sorted(df['Instrument_Base'].unique().tolist())
    selected_instrument = st.sidebar.selectbox("Filter by Instrument:", instruments)
    
    # Date range filter
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    date_range = st.sidebar.date_input(
        "Date Range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_instrument != 'All':
        filtered_df = filtered_df[filtered_df['Instrument_Base'] == selected_instrument]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['Date'].dt.date >= start_date) & 
            (filtered_df['Date'].dt.date <= end_date)
        ]
    
    # Main dashboard
    st.header("📈 Performance Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_net_pl = filtered_df['NetPl_Clean'].sum()
    total_trades = len(filtered_df)
    winning_trades = len(filtered_df[filtered_df['NetPl_Clean'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    avg_trade = filtered_df['NetPl_Clean'].mean()
    
    with col1:
        st.metric("Total Net P&L", f"${total_net_pl:,.2f}")
    with col2:
        st.metric("Total Trades", total_trades)
    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col4:
        st.metric("Avg. Trade P&L", f"${avg_trade:.2f}")
    
    # Equity Curve
    st.subheader("Equity Curve")
    fig_equity = px.line(
        filtered_df, 
        x='Date', 
        y='Cumulative_NetPL',
        title="Cumulative P&L Over Time"
    )
    fig_equity.update_layout(height=400)
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # P&L Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("P&L Distribution")
        fig_hist = px.histogram(
            filtered_df,
            x='NetPl_Clean',
            nbins=20,
            title="Distribution of Trade P&L"
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("Performance by Instrument")
        inst_pl = filtered_df.groupby('Instrument_Base')['NetPl_Clean'].sum().sort_values()
        fig_bar = px.bar(
            inst_pl,
            orientation='h',
            title="Total P&L by Instrument"
        )
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Trade Details
    st.subheader("Trade History")
    
    # Summary statistics
    st.write("**Trade Summary:**")
    win_df = filtered_df[filtered_df['NetPl_Clean'] > 0]
    loss_df = filtered_df[filtered_df['NetPl_Clean'] < 0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Winning Trades", len(win_df))
    with col2:
        st.metric("Losing Trades", len(loss_df))
    with col3:
        avg_win = win_df['NetPl_Clean'].mean() if len(win_df) > 0 else 0
        st.metric("Avg. Win", f"${avg_win:.2f}")
    with col4:
        avg_loss = loss_df['NetPl_Clean'].mean() if len(loss_df) > 0 else 0
        st.metric("Avg. Loss", f"${avg_loss:.2f}")
    
    # Raw data table
    st.subheader("Raw Trade Data")
    display_columns = ['Date', 'Action', 'Instrument', 'NetPl_Clean', 'Result']
    st.dataframe(
        filtered_df[display_columns].rename(columns={'NetPl_Clean': 'Net P&L'}),
        use_container_width=True
    )

if __name__ == "__main__":
    main()
