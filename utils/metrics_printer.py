# utils/metrics_printer.py

class MetricsPrinter:
    @staticmethod
    def format_metrics_table(prediction_metrics=None):
        """Create formatted tables of all metrics including prediction and trading metrics."""
        output = []

        # Print prediction metrics if available
        if prediction_metrics:
            output.extend([
                "\nPrediction Metrics:",
                "-" * 140
            ])
            for model_name, metrics in prediction_metrics.items():
                output.extend([f"{model_name:<20} MSE: {metrics['mse']:<15.5f} MAE: {metrics['mae']:<15.5f} SE: {metrics['se']:<16.5f} RRMSE: {metrics['relative_rmse']:<13.2%} RMAE: {metrics['relative_mae']:<10.2%} (numpy)"])

        return "\n".join(output) + "\n"

    @staticmethod
    def write_metrics(file_path, prediction_metrics=None):
        """Write all metrics to file."""
        formatted_table = MetricsPrinter.format_metrics_table(
            prediction_metrics)

        with open(file_path, 'a') as f:
            f.write(formatted_table)
            f.write("\n")

    @staticmethod
    def print_metrics(prediction_metrics=None):
        """Print all metrics to console."""
        formatted_table = MetricsPrinter.format_metrics_table(
            prediction_metrics)
        print(formatted_table)