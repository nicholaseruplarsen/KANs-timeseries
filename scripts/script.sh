#!/bin/bash

# Default values for parameters
seq_len=336
seeds=(421)
models=('KAN')
pred_lens=(18)
datasets=("MRO")
features_list=("S")

# Create necessary directories if they do not exist
if [ ! -d "./results" ]; then
    mkdir ./results
fi

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
            # Loop through each seed value
            for seed in "${seeds[@]}"; do
                # Loop through each feature type
                for features in "${features_list[@]}"; do
                    logname="${dataset}_${seq_len}_${pred_len}_${features}_channels_${channels}_seed_${seed}"
                    result_dir="./results/${logname}"

                    # Create result directory
                    if [ ! -d "$result_dir" ]; then
                        mkdir "$result_dir"
                    fi

                    # Create logs directory inside result directory
                    logs_dir="${result_dir}/logs"
                    if [ ! -d "$logs_dir" ]; then
                        mkdir "$logs_dir"
                    fi

                    log_file="${logs_dir}/${model}.log"
                    echo "Running experiment for seq_len=${seq_len}, pred_len=${pred_len}, seed=${seed} and feature=${features} ($log_file):"

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

                    python -u print_metrics.py
                done
            done
        done
    done
done