import numpy as np
import pyperclip

def format_latex_table(results_folder, datasets, seq_len, pred_lengths, models, mode):
    data = {}
    for model in models:
        data[model] = {}
        for pred_len in pred_lengths:
            data[model][pred_len] = {}
            for dataset in datasets:
                metrics = get_metrics(results_folder, dataset, seq_len, pred_len, model, models, mode)
                if metrics:
                    data[model][pred_len][dataset] = metrics
                else:
                    data[model][pred_len][dataset] = {'MSE': np.inf, 'MAE': np.inf, 'SE': np.inf, 'RRMSE': np.inf}

    # Define captions and labels based on mode
    captions = {
        'M': 'Multivariate input on multivariate columns',
        'MS': 'Multivariate input on univariate column',
        'S': 'Univariate input on univariate column'
    }
    labels = {
        'M': 'multivariate_multivariate',
        'MS': 'multivariate_univariate',
        'S': 'univariate_univariate'
    }

    latex_table = []
    latex_table.append(r"\begin{table*}[ht!]")
    latex_table.append(r"\centering")
    latex_table.append(r"\makebox[\textwidth][c]{")
    latex_table.append(r"\resizebox{1\textwidth}{!}{")
    latex_table.append(r"\begin{tabular}{@{}c|c|cccc|cccc|cccc|c@{}}")
    latex_table.append(r"\toprule")
    latex_table.append(r"Dataset & & \multicolumn{4}{c|}{Exchange Rate} & \multicolumn{4}{c|}{General Dynamics $\underline{GD}$} & \multicolumn{4}{c|}{Marathon Oil Corp $\underline{MRO}$} & Best \\ ")
    latex_table.append(r"\midrule")
    latex_table.append(r"Model & Horizon & MSE & MAE & SE & RRMSE & MSE & MAE & SE & RRMSE & MSE & MAE & SE & RRMSE & 48 \\ ")
    latex_table.append(r"\midrule")

    for model in models:
        latex_table.append(f"\\multirow{{4}}{{4em}}{{{model.replace('_', ' ')}}}")
        best_count = 0
        rows = []
        for pred_len in pred_lengths:
            row = f"& {pred_len} "
            for dataset in datasets:
                metrics = data[model][pred_len][dataset]
                for metric in ['MSE', 'MAE', 'SE', 'RRMSE']:
                    value = metrics[metric]
                    all_values = [data[m][pred_len][dataset][metric] for m in models]
                    sorted_values = sorted(set(all_values))
                    if metric == 'RRMSE':
                        if value == sorted_values[0]:
                            row += rf"& \textbf{{{value:.2f}\%}} "
                            best_count += 1
                        elif value == sorted_values[1]:
                            row += rf"& \underline{{{value:.2f}\%}} "
                        else:
                            row += f"& {value:.2f}\% "
                    else:
                        if value == sorted_values[0]:
                            row += rf"& \textbf{{{value:.3f}}} "
                            best_count += 1
                        elif value == sorted_values[1]:
                            row += rf"& \underline{{{value:.3f}}} "
                        else:
                            row += f"& {value:.3f} "
            rows.append(row)
        
        latex_table.append(rows[0] + rf"& \multirow{{4}}{{*}}{{{best_count}/48}} \\")
        for row in rows[1:]:
            latex_table.append(row + r"\\")
        latex_table.append(r"\midrule")

    latex_table.append(r"\end{tabular}")
    latex_table.append(r"}}")
    latex_table.append(rf"\caption{{{captions[mode]}}}")
    latex_table.append(rf"\label{{tab:{labels[mode]}}}")
    latex_table.append(r"\end{table*}")
    
    return "\n".join(latex_table)

def parse_metrics(file_content, models):
    metrics = {}
    for line in file_content.split('\n'):
        if 'MSE:' in line and any(model in line for model in models):
            parts = line.split()
            model_name = parts[0].strip()
            metrics[model_name] = {
                'MSE': float(parts[2]),
                'MAE': float(parts[4]),
                'SE': float(parts[6]),
                'RRMSE': float(parts[8]) * 100  # Convert RRMSE to percentage
            }
    return metrics

def get_metrics(results_folder, dataset, seq_len, pred_len, model, models, mode):
    channels = '8' if dataset == 'exchange_rate' else '6'
    model_file = 'Straight_Line.log' if model == 'Repeat' else f'{model}.log'
    log_file_path = f'{results_folder}/{dataset}_{seq_len}_{pred_len}_{mode}_channels_{channels}/logs/{model_file}'
    
    try:
        with open(log_file_path, 'r') as f:
            metrics = parse_metrics(f.read(), models)
            return metrics.get(model)
    except FileNotFoundError:
        print(f"Log file not found: {log_file_path}")
    return None

def print_metrics_table(results_folder, datasets, seq_len, pred_lengths, models, mode):
    headers = ['Dataset'] + [f"{dataset}\n(4 col)" for dataset in datasets]
    print(f"\nMode: {mode}")
    print(f"{headers[0]:<15} " + " ".join(f"{h:<40}" for h in headers[1:]))
    
    for model in models:
        print(f"\nProcessing model: {model}")
        print(f"{'Horizon':<7} " + " ".join(f"{'MSE':>10} {'MAE':>10} {'SE':>10} {'RRMSE':>10}" for _ in datasets))
        
        for pred_len in pred_lengths:
            row = [f"{pred_len:<7}"]
            for dataset in datasets:
                metrics = get_metrics(results_folder, dataset, seq_len, pred_len, model, models, mode)
                if metrics:
                    row.append(f"{metrics['MSE']:10.3f} {metrics['MAE']:10.3f} {metrics['SE']:10.3f} {metrics['RRMSE']:9.2f}%")
                else:
                    row.append(f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            print("".join(row))

# Example usage
# datasets = ['exchange_rate', 'GD', 'MRO']
datasets = ['AAPL', 'NVDA', 'TSLA']
seq_len = 336
pred_lengths = [96, 192, 336]
models = ['Repeat', 'DLinear']
results_folder = 'results'  # Change this to the path of your results folder

# Process all three modes
modes = ['S']
all_latex_tables = []

for mode in modes:
    print_metrics_table(results_folder, datasets, seq_len, pred_lengths, models, mode)
    print(f"\n\nLaTeX Table for mode {mode}:\n")
    latex_table = format_latex_table(results_folder, datasets, seq_len, pred_lengths, models, mode)
    print(latex_table)
    all_latex_tables.append(latex_table)

# Copy all tables to clipboard with newlines between them
combined_tables = "\n\n".join(all_latex_tables)
pyperclip.copy(combined_tables)
print("\nAll LaTeX tables have been copied to clipboard.")