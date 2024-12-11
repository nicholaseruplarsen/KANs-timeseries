import sys
import json
import numpy as np
from utils.metrics_printer import MetricsPrinter

def parse_metrics_file(file_path):
    """Parse a metrics file and extract the values."""
    metrics = {}
    with open(file_path, 'r') as f:
        content = f.read()
        # Parse the metrics using string manipulation
        for line in content.split('\n'):
            if line and not line.startswith('-'):
                parts = line.split()
                if len(parts) >= 11:  # Make sure we have enough parts
                    model_name = parts[0]
                    metrics[model_name] = {
                        'mse': float(parts[2]),
                        'mae': float(parts[4]),
                        'se': float(parts[6]),
                        'relative_rmse': float(parts[8].strip('%')) / 100,
                        'relative_mae': float(parts[10].strip('%')) / 100
                    }
    return metrics

def main():
    metrics_files = sys.argv[1:]
    all_runs_metrics = {}
    
    # Collect metrics from all runs
    for metrics_file in metrics_files:
        run_metrics = parse_metrics_file(f"./temp_metrics/{metrics_file}")
        for model_name, metrics in run_metrics.items():
            if model_name not in all_runs_metrics:
                all_runs_metrics[model_name] = []
            all_runs_metrics[model_name].append(metrics)
    
    # Print aggregated statistics
    MetricsPrinter.print_metrics(all_runs_metrics=all_runs_metrics)

if __name__ == "__main__":
    main()