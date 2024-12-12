# print_metrics.py
import os
import glob

def print_final_metrics():
    # Look for metrics files in temp_metrics directory
    metrics_files = sorted(glob.glob('temp_metrics/metrics_*.txt'))
    
    if metrics_files:
        # Print the metrics file corresponding to the current run
        current_seed = os.environ.get('CURRENT_SEED')  # We'll need to set this in script.sh
        if current_seed:
            current_file = f'temp_metrics/metrics_{current_seed}.txt'
            if os.path.exists(current_file):
                with open(current_file, 'r') as f:
                    print(f.read())
            else:
                print(f"\nMetrics file not found for seed {current_seed}")
        else:
            # Fallback to printing latest if seed not set
            latest_file = metrics_files[-1]
            with open(latest_file, 'r') as f:
                print(f.read())
    else:
        print("\nNo metrics files found.")

print_final_metrics()