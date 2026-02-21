
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import get_ticker_fundamentals
from src.storage import initialize_db, save_fundamentals, get_fundamentals

def verify():
    print("Initializing DB...")
    initialize_db()
    
    ticker = 'SHY'
    print(f"Fetching fundamentals for {ticker}...")
    fund = get_ticker_fundamentals(ticker)
    print(f"Fetched: {fund}")
    
    if fund['source'] == 'none':
        print("Warning: No source found (APIs might be limited or key missing).")
        # Creating mock data if API fails to strict test DB
        fund = {'dividend_yield': 0.042, 'pe_ratio': 0, 'market_cap': 1000000, 'source': 'mock_test'}
    
    print("Saving to DB...")
    save_fundamentals(ticker, fund)
    
    print("Reading from DB...")
    stored = get_fundamentals(ticker)
    print(f"Stored: {stored}")
    
    if abs(stored.get('dividend_yield', 0) - fund['dividend_yield']) < 0.0001:
        print("SUCCESS: Data retrieved matches data saved.")
    else:
        print("FAILURE: Data verification failed.")

if __name__ == "__main__":
    verify()
