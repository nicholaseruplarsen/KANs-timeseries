#!/bin/bash

# Default values for parameters
seq_len=252
seeds=(0 42 420 1337 2024)
models=('KAN')
pred_lens=(120)
datasets=("MRO")
features_list=("S")

# Create necessary directories if they do not exist
if [ ! -d "./results" ]; then
    mkdir ./results
fi

# Create a temporary directory for aggregated metrics
mkdir -p ./temp_metrics

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    # Set target and features based on dataset
    if [ "$dataset" = "exchange_rate" ]; then
        target="OT"
        channels=8
    else
        target="Adj Close"
        channels=6
    fi

    # Loop through each model
    for model in "${models[@]}"; do
        # Loop through each pred_len value
        for pred_len in "${pred_lens[@]}"; do
            # Loop through each feature type
            for features in "${features_list[@]}"; do
                # Create array to store metrics files
                metrics_files=()
                
                # Loop through each seed value
                for seed in "${seeds[@]}"; do
                    export CURRENT_SEED=$seed
                    # Clean up old metrics files
                    rm -f "./temp_metrics/metrics_${seed}.txt"
                    
                    logname="${dataset}_${seq_len}_${pred_len}_${features}_channels_${channels}_seed_${seed}"
                    result_dir="./results/${logname}"
                    
                    # Create result directory
                    mkdir -p "$result_dir/logs"
                    
                    log_file="${result_dir}/logs/${model}.log"
                    echo "Running experiment for seq_len=${seq_len}, pred_len=${pred_len}, seed=${seed} and feature=${features} ($log_file):"

                    # Run the experiment
                    python -u run_longExp.py \
                      --is_training 1 \
                      --root_path ./dataset/ \
                      --data_path "$dataset.csv" \
                      --model_id "${logname}" \
                      --model "$model" \
                      --data custom \
                      --features "$features" \
                      --target "$target" \
                      --seq_len $seq_len \
                      --pred_len $pred_len \
                      --enc_in $channels \
                      --des 'Exp' \
                      --seed $seed \
                      --save_npy 1 \
                      --itr 1 --batch_size 8 --learning_rate 0.0005 > "$log_file"

                    # Move and store metrics
                    mv final_metrics.txt "./temp_metrics/metrics_${seed}.txt"
                    metrics_files+=("metrics_${seed}.txt")
                    
                    # Print metrics for this run
                    python -u print_metrics.py
                done

                # Calculate and display aggregate statistics using the collected file paths
                python -u aggregate_metrics.py "${metrics_files[@]}"
                
                # Cleanup after aggregating
                rm -f ./temp_metrics/metrics_*.txt
            done
        done
    done
done