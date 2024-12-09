import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FuncFormatter

def dollar_formatter(x, p):
    return f'${x:.0f}'

def plot_dataset(filepath, ax, title):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Get total length and calculate split points
        total_len = len(df)
        train_len = int(total_len * 0.7)
        test_len = int(total_len * 0.2)
        val_len = total_len - train_len - test_len
        
        # Use dates for x-axis instead of indices
        dates = df['date']
        
        # Plot the full line in black
        ax.plot(dates, df.iloc[:, -1], 'k-', alpha=0.5, linewidth=1)
        
        # Fill colors for different sections
        ax.fill_between(dates[:train_len], 
                       df.iloc[:train_len, -1], 
                       alpha=0.2, 
                       color='green', 
                       label='Training')
        
        ax.fill_between(dates[train_len:train_len+val_len], 
                       df.iloc[train_len:train_len+val_len, -1], 
                       alpha=0.2, 
                       color='purple', 
                       label='Validation')
        
        ax.fill_between(dates[train_len+val_len:], 
                       df.iloc[train_len+val_len:, -1], 
                       alpha=0.2, 
                       color='blue', 
                       label='Test')
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        
        # Format y-axis with dollar signs for stock data
        if 'GD' in filepath or 'MRO' in filepath:
            ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    else:
        ax.text(0.5, 0.5, f'File not found: {filepath}\nCWD: {os.getcwd()}', 
               ha='center', va='center')
        ax.set_title(title)

def main():
    # Use seaborn style for better looking plots
    # plt.style.use('seaborn-v0_8')
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    
    # Plot each dataset
    data_files = [
        ('dataset/AAPL.csv', 'AAPL'),
        ('dataset/NVDA.csv', 'NVDA'),
        ('dataset/TSLA.csv', 'TSLA')
    ]
    
    for (filename, title), ax in zip(data_files, [ax1, ax2, ax3]):
        plot_dataset(filename, ax, title)
    
    # Adjust layout with more space at bottom for rotated dates
    plt.tight_layout()
    
    # Save the plot
    # plt.savefig('dataset_splits.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()