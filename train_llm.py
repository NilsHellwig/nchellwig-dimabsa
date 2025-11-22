from llm import train_and_evaluate
import sys
import os
import json
import random
import argparse
import logging

import sys
from helper import *

logger = logging.getLogger('training')


def run_training_pipeline_real(subtask=3, language="eng", domain="restaurant", seed_run=0, strategy="pred_dev"):

    if strategy == "pred_dev":
        train_data = get_dataset(
            subtask=subtask, language=language, domain=domain, split="train")
        if train_data is None:
            logger.warning(f"Dataset not found for subtask={subtask}, language={language}, domain={domain}, split=train. Skipping.")
            return None
        
        test_dataset = get_dataset(
            subtask=subtask, language=language, domain=domain, split="dev")
        if test_dataset is None:
            logger.warning(f"Dataset not found for subtask={subtask}, language={language}, domain={domain}, split=dev. Skipping.")
            return None
    elif strategy == "train_split":
        dataset = get_dataset(
            subtask=subtask, language=language, domain=domain, split="train")
        if dataset is None:
            logger.warning(f"Dataset not found for subtask={subtask}, language={language}, domain={domain}, split=train. Skipping.")
            return None
        
        test_dataset = dataset[:500]
        train_data = dataset[500:]

    results = train_and_evaluate(
        train_data,
        test_dataset,
        subtask=subtask,
        language=language,
        domain=domain,
        seed=seed_run,
        num_train_epochs=5
    )

    # delete model_temp and free trash
    import shutil
    model_temp_path = "./model_temp"
    if os.path.exists(model_temp_path):
        shutil.rmtree(model_temp_path)
        logger.info(f"Deleted {model_temp_path}")

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
    if strategy == "pred_dev":
        results_dir = f"results/results_pred_dev/subtask_{subtask}"
        os.makedirs(results_dir, exist_ok=True)
        path_predictions = f"{results_dir}/pred_{language}_{domain}.jsonl"
    else:
        os.makedirs("results", exist_ok=True)
        path_predictions = f"results/predictions_{subtask}_{language}_{domain}_{seed_run}_{strategy}.jsonl"

    logger.info(f"="*80)
    logger.info(f"Starting experiment: subtask={subtask}, language={language}, domain={domain}, seed={seed_run}, strategy={strategy}")
    
    if os.path.exists(path_predictions):
        logger.info(
            f"Predictions for {subtask} {language} {domain} seed {seed_run} already exist. Skipping.")
        return

    predictions = run_training_pipeline_real(subtask, language, domain, seed_run, strategy)

    # If predictions is None, dataset doesn't exist - skip
    if predictions is None:
        logger.warning(f"Skipping {subtask} {language} {domain} seed {seed_run} - dataset not found.")
        return

    # Save predictions in JSONL format
    with open(path_predictions, "w") as f_out:
        for pred in predictions:
            f_out.write(json.dumps(pred) + "\n")

    logger.info(f"Predictions saved to {path_predictions}")
    logger.info(f"Experiment completed successfully")
    logger.info(f"="*80)


if __name__ == "__main__":
    main()
