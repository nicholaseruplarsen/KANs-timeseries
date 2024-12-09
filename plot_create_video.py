import cv2
from plot_utils import configure, load_data, load_naive_pred, plot_and_save_frame, prepare_plot_data
import numpy as np
import os
import glob

def extract_folder_info(folder_path):
    """Extract dataset, seq_len, pred_len, and seed from folder name."""
    base = os.path.basename(folder_path)
    parts = base.split('_')
    
    # Handle cases with and without seed
    dataset = parts[0]
    seq_len = int(parts[1])
    pred_len = int(parts[2])
    seed = None
    
    # Check if seed exists in folder name
    if 'seed' in base:
        seed = int(parts[-1])  # Get the last part which should be the seed number
        
    return dataset, seq_len, pred_len, seed

def find_matching_results_folder(base_path, dataset, seq_len, pred_len):
    """Find all matching results folders, including those with seeds."""
    pattern = f"{dataset}_{seq_len}_{pred_len}_S_channels_*"
    matching_folders = glob.glob(os.path.join(base_path, pattern))
    
    if not matching_folders:
        print(f"No matching folders found for pattern: {pattern}")
        return None
        
    # If multiple folders exist, prefer the one without seed, otherwise take the first one
    for folder in matching_folders:
        if 'seed' not in folder:
            return folder
    
    return matching_folders[0]

def setup_video(root_folder, specified_model, total_samples, dim_to_plot, video_duration=5):
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # or try 'H264', 'XVID', 'MJPG'
    
    # Extract folder info to include in video name
    dataset, seq_len, pred_len, seed = extract_folder_info(root_folder)
    
    # Create video filename with seed information if it exists
    if seed is not None:
        video_name = f'forecast_video_{specified_model}_{total_samples}_samples_dim_{dim_to_plot+1}_seed_{seed}.mp4'
    else:
        video_name = f'forecast_video_{specified_model}_{total_samples}_samples_dim_{dim_to_plot+1}.mp4'
        
    video_path = os.path.join(root_folder, video_name)
    return fourcc, video_path, None, video_duration

def create_forecast_video(dataset, seq_len, pred_len, specified_model, dim_to_plot, sample_interval, root_folder, model_list, video_duration=5, create_ensemble=True, ensemble_models=None):
    dim_to_plot = dim_to_plot - 1
    
    # Find the correct results folder
    if not os.path.exists(root_folder):
        root_folder = find_matching_results_folder('results', dataset, seq_len, pred_len)
        if root_folder is None:
            print("Could not find matching results folder")
            return
    
    root_folder, input_len, models, exclude_repeat = configure(dataset, seq_len, pred_len, root_folder, model_list)
    
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
    print(f"Total samples: {total_samples}")
    fourcc, video_path, video, video_duration = setup_video(root_folder, specified_model, total_samples, dim_to_plot, video_duration)

    # Calculate frames per second based on number of samples and desired duration
    total_frames = total_samples // sample_interval + 1
    fps = total_frames / video_duration

    for sample in range(0, total_samples, sample_interval):
        gt_data = gt[sample]
        
        results = prepare_plot_data(gt_data, data, naive_pred, pred_len, dim_to_plot, sample, create_ensemble=create_ensemble, ensemble_models=ensemble_models)
        
        if 'Ensemble' in results:
            models['Ensemble'] = {'color': 'purple', 'style': '-'}

        frame = plot_and_save_frame(gt_data, results, models, input_len, sample, total_samples, specified_model, exclude_repeat, dim_to_plot, dataset)

        if video is None:
            h, w = frame.shape[:2]
            video = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        
        video.write(frame)
        print(f"Processed sample {sample+1}/{total_samples+1}")

    if video is not None:
        video.release()
    print(f"Video saved to {video_path} with duration {video_duration} seconds at {fps:.2f} fps")

if __name__ == "__main__":
    dataset = 'MROh'
    seq_len = 336
    pred_len = 40
    specified_model = 'DLinear'
    model_list = ['DLinear']
    
    root_folder = f'results/{dataset}_{seq_len}_{pred_len}_S_channels_6_seed_2021'

    dim_to_plot = 1
    sample_interval = 5
    video_duration = 10
    
    create_forecast_video(dataset, seq_len, pred_len, specified_model, dim_to_plot, sample_interval, root_folder, model_list, video_duration, create_ensemble=False)