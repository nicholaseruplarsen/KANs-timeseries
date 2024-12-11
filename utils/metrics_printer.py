# utils/metrics_printer.py
import numpy as np
from scipy import stats

class MetricsPrinter:
    @staticmethod
    def format_metrics_table(prediction_metrics=None, all_runs_metrics=None):
        """Create formatted tables of metrics including single run and aggregated statistics."""
        output = []

        # Print prediction metrics if available for current run
        if prediction_metrics:
            output.extend([
                "\nPrediction Metrics:",
                "-" * 140
            ])
            for model_name, metrics in prediction_metrics.items():
                output.extend([f"{model_name:<20} MSE: {metrics['mse']:<15.5f} MAE: {metrics['mae']:<15.5f} SE: {metrics['se']:<16.5f} RRMSE: {metrics['relative_rmse']:<13.2%} RMAE: {metrics['relative_mae']:<10.2%} (numpy)"])

        # Print aggregated statistics if available
        if all_runs_metrics:
            output.extend([
                "\nAggregated Statistics Across All Runs:",
                "-" * 140
            ])
            
            models = list(all_runs_metrics.keys())
            if len(models) >= 2:  # If we have at least two models to compare
                base_model, comparison_model = models[0], models[1]
                
                # Perform paired t-test for MSE
                mse_ttest = stats.ttest_rel(
                    [run['mse'] for run in all_runs_metrics[base_model]], 
                    [run['mse'] for run in all_runs_metrics[comparison_model]]
                )
                
                for model_name, runs in all_runs_metrics.items():
                    mse_values = [run['mse'] for run in runs]
                    mae_values = [run['mae'] for run in runs]
                    se_values = [run['se'] for run in runs]
                    rrmse_values = [run['relative_rmse'] for run in runs]
                    rmae_values = [run['relative_mae'] for run in runs]
                    
                    output.extend([
                        f"{model_name:<20} "
                        f"MSE: {np.mean(mse_values):<10.5f}±{np.std(mse_values):<10.5f} "
                        f"MAE: {np.mean(mae_values):<10.5f}±{np.std(mae_values):<10.5f} "
                        f"SE: {np.mean(se_values):<10.5f}±{np.std(se_values):<10.5f} "
                        f"RRMSE: {np.mean(rrmse_values):.2%}±{np.std(rrmse_values):.2%} "
                        f"RMAE: {np.mean(rmae_values):.2%}±{np.std(rmae_values):.2%}"
                    ])
                
                output.extend([
                    f"\nStatistical Tests:",
                    f"Paired t-test (MSE) p-value: {mse_ttest.pvalue:.5f}"
                ])

        return "\n".join(output) + "\n"

    @staticmethod
    def write_metrics(file_path, prediction_metrics=None, all_runs_metrics=None):
        """Write all metrics to file."""
        formatted_table = MetricsPrinter.format_metrics_table(
            prediction_metrics, all_runs_metrics)

        with open(file_path, 'a') as f:
            f.write(formatted_table)
            f.write("\n")

    @staticmethod
    def print_metrics(prediction_metrics=None, all_runs_metrics=None):
        """Print all metrics to console."""
        formatted_table = MetricsPrinter.format_metrics_table(
            prediction_metrics, all_runs_metrics)
        print(formatted_table)