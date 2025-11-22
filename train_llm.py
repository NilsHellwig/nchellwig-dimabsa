from llm import train_and_evaluate
import sys
import os
import json
import random
import argparse

import sys
from helper import *


def run_training_pipeline_real(subtask=3, language="eng", domain="restaurant", seed_run=0, strategy="pred_dev"):

    if strategy == "pred_dev":
        train_data = get_dataset(
            subtask=subtask, language=language, domain=domain, split="train")
        test_dataset = get_dataset(
            subtask=subtask, language=language, domain=domain, split="dev")
    elif strategy == "train_split":
        test_dataset = get_dataset(
            subtask=subtask, language=language, domain=domain, split="train")[:500]
        train_data = get_dataset(
            subtask=subtask, language=language, domain=domain, split="train")[500:]

    results = train_and_evaluate(
        train_data,
        test_dataset,
        subtask=subtask,
        domain=domain,
        seed=seed_run,
        num_train_epochs=5
    )

    # delete model_temp and free trash
    import shutil
    model_temp_path = "./model_temp"
    if os.path.exists(model_temp_path):
        shutil.rmtree(model_temp_path)
        print(f"Deleted {model_temp_path}")

    clear_memory()

    return results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run LLM training and evaluation')
    parser.add_argument('--subtask', type=int, help='Subtask number')
    parser.add_argument('--language', type=str, help='Language code')
    parser.add_argument('--domain', type=str, help='Domain name')
    parser.add_argument('--seed_run', type=int, help='Seed run number')
    parser.add_argument('--strategy', type=str, default="pred_dev")
    args = parser.parse_args()

    setup_gpu_environment()
    clear_memory()

    subtask = args.subtask
    language = args.language
    domain = args.domain
    seed_run = args.seed_run
    strategy = args.strategy

    # Erstelle results Ordner falls nicht vorhanden
    os.makedirs("results", exist_ok=True)

    path_predictions = f"results/predictions_{subtask}_{language}_{domain}_{seed_run}_{strategy}.jsonl"

    if os.path.exists(path_predictions):
        print(
            f"Predictions for {subtask} {language} {domain} seed {seed_run} already exist. Skipping.")
        return

    predictions = run_training_pipeline_real(subtask, language, domain, seed_run, strategy)

    # Save predictions in JSONL format
    with open(path_predictions, "w") as f_out:
        for pred in predictions:
            f_out.write(json.dumps(pred) + "\n")

    print(f"Predictions saved to {path_predictions}")


if __name__ == "__main__":
    main()
