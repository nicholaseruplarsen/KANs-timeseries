# plot_single_frame.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
import cv2
from utils.metrics import MSE, MAE, SE, RMSE
import matplotlib.pyplot as plt
import numpy as np

def create_ensemble_prediction(results, dim_to_plot):
    """
    Creates an ensemble prediction by averaging all model predictions for the specified dimension.
    """
    # Get the shape of predictions from the first model
    first_model = next(iter(results.values()))
    pred_length = first_model['pd_data'].shape[0]
    
    # Initialize array to store all predictions for the specified dimension
    all_predictions = np.zeros((len(results), pred_length))
    
    # Stack all predictions for the specified dimension
    for i, (model_name, model_data) in enumerate(results.items()):
        all_predictions[i] = model_data['pd_data'][:, dim_to_plot]
    
    # Calculate ensemble prediction (simple average)
    ensemble_pred = np.mean(all_predictions, axis=0)
    
    # Reshape ensemble prediction to match the expected format (pred_length, 6)
    final_pred = np.zeros((pred_length, 6))
    final_pred[:, dim_to_plot] = ensemble_pred
    
    # Calculate ensemble metrics
    ensemble_mse = np.mean([model_data['mse'] for model_data in results.values()])
    ensemble_se = np.mean([model_data['se'] for model_data in results.values()])
    
    return {
        'mse': ensemble_mse,
        'se': ensemble_se,
        'pd_data': final_pred
    }

def prepare_plot_data(gt_data, data, naive_pred, pred_len, dim_to_plot, sample_index, 
                     create_ensemble=True, ensemble_models=None):
    """
    Centralized function to prepare data for plotting
    
    Args:
        ... (existing args) ...
        create_ensemble (bool): Whether to create an ensemble prediction
        ensemble_models (list): List of model names to include in ensemble. If None, uses all models.
    """
    results = {}
    
    # Process model predictions
    for model, model_data in data.items():
        pd, _ = model_data
        if pd is not None:
            pd_data = pd[sample_index]
            gt_slice = gt_data[-pred_len:, dim_to_plot]
            pd_slice = pd_data[:, dim_to_plot]
            mse = np.mean((gt_slice - pd_slice)**2)
            se = np.mean((gt_data[-1, dim_to_plot] - pd_data[-1, dim_to_plot])**2)
            results[model] = {'mse': mse, 'se': se, 'pd_data': pd_data}

    # Add naive prediction if available
    if naive_pred is not None:
        naive_pred_data = naive_pred[sample_index]
        gt_slice = gt_data[-pred_len:, dim_to_plot]
        naive_slice = naive_pred_data[:, dim_to_plot]
        mse_naive = np.mean((gt_slice - naive_slice)**2)
        se_naive = np.mean((gt_data[-1, dim_to_plot] - naive_pred_data[-1, dim_to_plot])**2)
        results['Repeat'] = {'mse': mse_naive, 'se': se_naive, 'pd_data': naive_pred_data}

    # Add ensemble prediction if enabled and we have models to ensemble
    if create_ensemble and len(results) > 1:
        # Filter models for ensemble if specified
        ensemble_dict = results
        if ensemble_models is not None:
            ensemble_dict = {k: v for k, v in results.items() if k in ensemble_models}
            
        if len(ensemble_dict) > 1:  # Only create ensemble if we have at least 2 models
            results['Ensemble'] = create_ensemble_prediction(ensemble_dict, dim_to_plot)
    
    return results

def configure(dataset, seq_len, pred_len, root_folder, model_list):
    cmap = plt.get_cmap('viridis')
    colors = [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0.3, 0.9, len(model_list))]
    models = {'Repeat': {'color': 'gray', 'style': '--'}}
    models.update({model: {'color': color, 'style': '-'} for model, color in zip(model_list, colors)})
    exclude_repeat = False
    return root_folder, seq_len, models, exclude_repeat

def load_data(root_folder, model_name):
    pd_path = os.path.join(root_folder, f'{model_name}_pred.npy')
    metric_file_names = [
        f'{model_name}_metrics.npy',
        f'{"_".join(model_name.split("_")[::-1])}_metrics.npy',
        f'{"_".join(model_name.split("_"))}_metrics.npy'
    ]
    
    metrics_path = None
    for file_name in metric_file_names:
        path = os.path.join(root_folder, file_name)
        if os.path.exists(path):
            metrics_path = path
            break
    
    if os.path.exists(pd_path):
        pd = np.load(pd_path)
        metrics = np.load(metrics_path, allow_pickle=True).item() if metrics_path else None
        return pd, metrics
    else:
        print(f"Data not found for {model_name} in {root_folder}")
        return None, None

def load_naive_pred(root_folder):
    gt_path = os.path.join(root_folder, 'gt.npy')
    naive_pred_path = os.path.join(root_folder, 'naive_pred.npy')
    
    if os.path.exists(gt_path) and os.path.exists(naive_pred_path):
        gt = np.load(gt_path)
        naive_pred = np.load(naive_pred_path)
        return gt, naive_pred
    else:
        print(f"Ground truth or naive prediction file not found in {root_folder}")
        return None, None

def create_plot(gt_data, results, models, input_len, sample, total_samples, specified_model, exclude_repeat, dim_to_plot, dataset):
    """Creates the standardized plot and returns the figure and axes objects"""
    fig = plt.figure(figsize=(12, 6), dpi=100)
    ax = plt.gca()
    
    x_gt = range(len(gt_data))
    y_gt = gt_data[:, dim_to_plot]
    
    # Calculate y limits
    y_min = min(np.min(gt_data[:, dim_to_plot]), min(np.min(data['pd_data'][:, dim_to_plot]) for data in results.values()))
    y_max = max(np.max(gt_data[:, dim_to_plot]), max(np.max(data['pd_data'][:, dim_to_plot]) for data in results.values()))
    y_range = y_max - y_min
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    # Create gradient background
    cmap = plt.get_cmap('YlGnBu_r')
    xv, yv = np.meshgrid(np.linspace(0, len(x_gt)-1, 100), np.linspace(y_min, y_max, 100))
    zv = yv
    plt.imshow(zv, cmap=cmap, origin='upper', aspect='auto',
               extent=[0, len(x_gt)-1, y_min, y_max], alpha=0.1)

    # Create first legend handles for regions and ground truth
    legend1_handles = []
    
    # Fill regions for input and forecast
    lookback = plt.fill_between(range(input_len), y_gt[:input_len], y_min, 
                    color='green', alpha=0.05)
    prediction = plt.fill_between(range(input_len, len(x_gt)), y_gt[input_len:], y_min,
                    color='blue', alpha=0.05)

    # Plot ground truth
    plt.fill_between(x_gt, y_gt, y_max, color='white', alpha=0.7, zorder=5)
    ground_truth = plt.plot(x_gt, y_gt, linewidth=1.8, color='black', zorder=6)[0]
    
    legend1_handles.extend([ground_truth, lookback, prediction])

    # Create first legend
    first_legend = plt.legend(legend1_handles, 
                            ['Ground Truth', 'Lookback Window', 'Prediction'],
                            loc='lower left', bbox_to_anchor=(0, 0))
    
    # Add the first legend manually to the plot
    ax.add_artist(first_legend)

    # Create second legend for model predictions
    legend2_handles = []
    legend2_labels = []

    # Plot predictions
    for model, data in results.items():
        if model != 'Repeat' or not exclude_repeat:
            label = f'{model} [MSE: {data["mse"]:.3f}]'
            forecast_data = data["pd_data"][:, dim_to_plot]
            x_forecast = range(input_len, input_len + len(forecast_data))
            
            alpha = 0.5 if 'DLinear' in model else 1.0
            
            line = plt.plot(x_forecast, forecast_data, linewidth=1.2, 
                    color=models[model]['color'], linestyle=models[model]['style'], 
                    zorder=7, alpha=alpha)[0]
            
            plt.scatter(x_forecast[-1], forecast_data[-1], color=models[model]['color'], 
                       s=50, zorder=8)
            plt.annotate(f'SE: {data["se"]:.2f}', (x_forecast[-1], forecast_data[-1]), 
                        xytext=(-26, 5), textcoords='offset points', 
                        color=models[model]['color'], zorder=9)
            
            legend2_handles.append(line)
            legend2_labels.append(label)

    plt.axvline(x=input_len, color='silver', linestyle='--', zorder=4)

    # Create second legend
    plt.legend(legend2_handles, legend2_labels, 
              loc='upper left', bbox_to_anchor=(0, 1))

    # Set title
    sorted_models = sorted([(k, v) for k, v in results.items() if k != 'Repeat' or not exclude_repeat], 
                         key=lambda x: x[1]['mse'])
    
    if len(sorted_models) >= 2:
        best_model, second_best_model = sorted_models[0][0], sorted_models[1][0]
        compared_model = second_best_model if specified_model == best_model else best_model
        mse_diff = results[compared_model]['mse'] - results[specified_model]['mse']
        title = f'{specified_model} {"better" if mse_diff > 0 else "worse"} than {compared_model} by {"-" if mse_diff < 0 else "+"}{abs(mse_diff):.3f} MSE.\nSample {sample+1}/{total_samples} on ${dataset}. (Dim: {dim_to_plot+1})'
    else:
        title = f'{specified_model} MSE: {results[specified_model]["mse"]:.3f}.\nSample {sample+1}/{total_samples}, Dim {dim_to_plot+1}'

    plt.title(title)
    
    # Format axes
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Stock price')
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    return fig

def plot_and_save_frame(gt_data, results, models, input_len, sample, total_samples, specified_model, exclude_repeat, dim_to_plot, dataset):
    fig = create_plot(gt_data, results, models, input_len, sample, total_samples, 
                     specified_model, exclude_repeat, dim_to_plot, dataset)
    
    # Convert to video frame
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)

    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()

    # Remove the RGB to BGR conversion since we want to preserve the original colors
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Remove this line
    return frame


def plot_single_frame(dataset, seq_len, pred_len, specified_model, sample_index, dim_to_plot, root_folder, model_list, create_ensemble=True, ensemble_models=None):
    dim_to_plot = dim_to_plot - 1
    sample_index -= 1
    root_folder, input_len, models, exclude_repeat = configure(dataset, seq_len, pred_len, root_folder, model_list)
    
    # Load data
    gt, naive_pred = load_naive_pred(root_folder)
    data = {}
    for model in models.keys():
        if model != 'Repeat':
            pd, metrics = load_data(root_folder, model)
            if gt is not None and pd is not None:
                data[model] = (pd, metrics)
    
    if not data:
        print(f"No data found for any model in {root_folder}")
        return

    total_samples = gt.shape[0]
    if sample_index >= total_samples:
        print(f"Sample index {sample_index+1} is out of range. Total samples: {total_samples}")
        return
    
    print(f"Plotting index: {sample_index+1}/{total_samples}")

    gt_data = gt[sample_index]
    
    # Prepare plot data using centralized function
    results = prepare_plot_data(gt_data, data, naive_pred, pred_len, dim_to_plot, sample_index, create_ensemble=create_ensemble, ensemble_models=ensemble_models)
    
    # Add ensemble to models dictionary if it exists
    if 'Ensemble' in results:
        models['Ensemble'] = {'color': 'purple', 'style': '-'}
    
    # Create and display plot
    fig = create_plot(gt_data, results, models, input_len, sample_index, total_samples, 
                     specified_model, exclude_repeat, dim_to_plot, dataset)
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    dataset = 'GD'
    seq_len = 336
    pred_len = 192
    specified_model = 'DLinear'
    dim_to_plot = 1                 # 1-based indexing
    sample_index = 1001             # 1-based indexing
    root_folder = f'results/old/{dataset}_{seq_len}_{pred_len}_S_channels_6'
    model_list = ['' '']
    
    # Example 2: Ensemble with specific models only
    ensemble_models = ['' '']
    plot_single_frame(dataset, seq_len, pred_len, specified_model, sample_index, dim_to_plot, root_folder, model_list, create_ensemble=True, ensemble_models=ensemble_models)