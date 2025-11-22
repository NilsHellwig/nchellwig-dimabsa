#!/bin/bash

SUBTASKS=(3 2)
LANGUAGES=("eng" "jpn" "rus" "tat" "ukr" "zho")
DOMAINS=("restaurant" "laptop" "hotel" "finance")
N_SEEDS_RUNS=1
STRATEGY=("pred_dev") # "train_split"

for seed_run in $(seq 0 $((N_SEEDS_RUNS - 1))); do
    for language in "${LANGUAGES[@]}"; do
      for subtask in "${SUBTASKS[@]}"; do
        for strategy in "${STRATEGY[@]}"; do
            for domain in "${DOMAINS[@]}"; do
                python train_llm.py \
                    --subtask "$subtask" \
                    --language "$language" \
                    --domain "$domain" \
                    --seed_run "$seed_run"
            done
        done
      done
    done
done