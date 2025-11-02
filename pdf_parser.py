import re
import pandas as pd
from datetime import datetime
import fitz  # PyMuPDF

class Plus500Parser:
    def __init__(self):
        self.trade_patterns = {
            'trade_date': r'(\d{2}/\d{2}/\d{4})',
            'instrument': r'(\d+\s+\d+\s+.*?Futures)',
            'exchange': r'(CBOT|CME|NYMEX|COMEX)',
            'pnl': r'PNL USD ([\(\)\d\.]+)\*',  # Updated to capture brackets
            'commission': r'FEE/COMM USD ([\(\)\d\.]+)\*'  # Updated for brackets
        }
    
    def parse_pdf(self, pdf_path):
        """Parse Plus500 PDF statement and extract trades"""
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        return self._extract_trades(full_text)
    
    def _extract_trades(self, text):
        """Extract trade data from statement text"""
        trades = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for trade date pattern
            date_match = re.match(r'(\d{2}/\d{2}/\d{4})', line)
            if date_match:
                trade_date = date_match.group(1)
                
                # Look for instrument in current and next lines
                instrument = ""
                j = i
                while j < min(i + 3, len(lines)):
                    if "Futures" in lines[j]:
                        instrument = self._clean_instrument(lines[j])
                        break
                    j += 1
                
                # Look for PNL and Commission in subsequent lines
                pnl = 0.0
                commission = 0.0
                exchange = "UNKNOWN"
                
                for k in range(i, min(i + 5, len(lines))):
                    line_k = lines[k]
                    
                    # Extract PNL with bracket handling
                    pnl_match = re.search(r'PNL USD ([\(\)\d\.]+)\*', line_k)
                    if pnl_match:
                        pnl_str = pnl_match.group(1)
                        pnl = self._parse_amount(pnl_str)
                        
                        # Determine exchange from the PNL line
                        if 'CBOT' in line_k:
                            exchange = 'CBOT'
                        elif 'CME' in line_k:
                            exchange = 'CME'
                    
                    # Extract Commission with bracket handling
                    comm_match = re.search(r'FEE/COMM USD ([\(\)\d\.]+)\*', line_k)
                    if comm_match:
                        commission_str = comm_match.group(1)
                        commission = self._parse_amount(commission_str)
                
                if instrument and (pnl != 0 or commission != 0):
                    trade = {
                        'trade_date': datetime.strptime(trade_date, '%m/%d/%Y'),
                        'instrument': instrument,
                        'exchange': exchange,
                        'pnl': pnl,
                        'commission': commission,
                        'net_pnl': pnl + commission,
                        'direction': 'LONG' if pnl > 0 else 'SHORT',
                        'quantity': self._extract_quantity(instrument)
                    }
                    trades.append(trade)
            
            i += 1
        
        return pd.DataFrame(trades)
    
    def _parse_amount(self, amount_str):
        """Parse amount string with bracket notation for losses"""
        # Remove any asterisks and strip whitespace
        amount_str = amount_str.replace('*', '').strip()
        
        # Check if it's in brackets (indicating negative/liability)
        if amount_str.startswith('(') and amount_str.endswith(')'):
            # Remove brackets and parse as negative number
            number_str = amount_str[1:-1]
            return -float(number_str)
        else:
            # Parse as positive number
            return float(amount_str)
    
    def _clean_instrument(self, instrument_text):
        """Clean and standardize instrument names"""
        # Remove quantity numbers and clean up
        cleaned = re.sub(r'^\d+\s+\d+\s+', '', instrument_text)
        cleaned = re.sub(r'Futures.*', 'Futures', cleaned)
        return cleaned.strip()
    
    def _extract_quantity(self, instrument_text):
        """Extract quantity from instrument text"""
        match = re.search(r'^(\d+)\s+\d+\s+', instrument_text)
        return int(match.group(1)) if match else 1
