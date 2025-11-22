#!/bin/bash

SUBTASKS=(2 3)
LANGUAGES=("eng")
DOMAINS_ENG=("laptop" "restaurant")
DOMAINS_OTHER=("restaurant" "laptop")  # Für andere Sprachen nur restaurant
STRATEGY=("pred_dev" "train_split")
N_SEEDS_RUNS=5

for seed_run in $(seq 0 $((N_SEEDS_RUNS - 1))); do
    for strategy in "${STRATEGY[@]}"; do
      for subtask in "${SUBTASKS[@]}"; do
        for language in "${LANGUAGES[@]}"; do
            # Bestimme verfügbare Domains basierend auf Sprache
            if [ "$language" == "eng" ]; then
                DOMAINS=("${DOMAINS_ENG[@]}")
            else
                DOMAINS=("${DOMAINS_OTHER[@]}")
            fi
            
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