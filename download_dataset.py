import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# Create dataset directory if it doesn't exist
if not os.path.exists('./dataset'):
    os.makedirs('./dataset')

tickers = ['AAPL', 'GD', 'MRO', 'NVDA', 'TSLA']

# Calculate start date (730 days ago)
end_date = datetime.now()
start_date = end_date - timedelta(days=729)

for ticker in tickers:
    # Download data using yf.download() with date range
    data = yf.download(ticker, 
                      start=start_date,
                      end=end_date,
                      interval="60m")
    
    # Reset index and rename Date column
    data.reset_index(inplace=True)
    data.rename(columns={'Datetime': 'date'}, inplace=True)  # Note: changed from 'Date' to 'Datetime'
    
    # Swap Volume and Adj Close columns
    cols = list(data.columns)
    vol_index = cols.index('Volume')
    adj_close_index = cols.index('Adj Close')
    cols[vol_index], cols[adj_close_index] = cols[adj_close_index], cols[vol_index]
    data = data[cols]
    
    # Save with 'h' suffix
    csv_path = f'./dataset/{ticker}h.csv'
    data.to_csv(csv_path, index=False)
    
    print(f"{ticker} hourly data saved to {csv_path}")