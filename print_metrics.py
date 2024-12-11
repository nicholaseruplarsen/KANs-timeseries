import os
import glob

def print_final_metrics():
    # Look for metrics files in temp_metrics directory
    metrics_files = sorted(glob.glob('temp_metrics/metrics_*.txt'))
    
    if metrics_files:
        # Just print the most recent metrics file
        latest_file = metrics_files[-1]
        with open(latest_file, 'r') as f:
            print(f.read())
    else:
        print("\nNo metrics files found.")

print_final_metrics()