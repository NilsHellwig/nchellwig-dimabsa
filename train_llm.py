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


def run_training_pipeline_real(subtask=3, language="eng", domain="restaurant", seed_run=0, strategy="pred_dev", split_idx=0, llm_name="unsloth/gemma-3-27b-it-bnb-4bit", num_epochs=5):

    if strategy == "pred_dev":
        train_dataset = get_dataset(
            subtask=subtask, language=language, domain=domain, split="train")
        if train_dataset is None:
            logger.warning(
                f"Dataset not found for subtask={subtask}, language={language}, domain={domain}, split=train. Skipping.")
            return None

        test_dataset = get_dataset(
            subtask=subtask, language=language, domain=domain, split="dev")
        if test_dataset is None:
            logger.warning(
                f"Dataset not found for subtask={subtask}, language={language}, domain={domain}, split=dev. Skipping.")
            return None
    elif strategy == "pred_test":
        train_dataset = get_dataset(
            subtask=subtask, language=language, domain=domain, split="train")
        if train_dataset is None:
            logger.warning(
                f"Dataset not found for subtask={subtask}, language={language}, domain={domain}, split=train. Skipping.")
            return None

        test_dataset = get_dataset(
            subtask=subtask, language=language, domain=domain, split="test")
        if test_dataset is None:
            logger.warning(
                f"Dataset not found for subtask={subtask}, language={language}, domain={domain}, split=test. Skipping.")
            return None
    elif strategy == "train_split":
        dataset = get_dataset(
            subtask=subtask, language=language, domain=domain, split="train")
        if dataset is None:
            logger.warning(
                f"Dataset not found for subtask={subtask}, language={language}, domain={domain}, split=train. Skipping.")
            return None

        # do a train/validation split (80/20)
        random.seed(seed_run)
        random.shuffle(dataset)
        # split in five different 80/20 splits
        splits = []
        split_size = len(dataset) // 5
        for i in range(5):
            start_idx = i * split_size
            if i == 4:  # last split takes the remainder
                end_idx = len(dataset)
            else:
                end_idx = (i + 1) * split_size
            splits.append(dataset[start_idx:end_idx])
        val_data = splits[split_idx]
        train_dataset = [item for i, split in enumerate(
            splits) if i != split_idx for item in split]
        test_dataset = val_data

    results = train_and_evaluate(
        train_dataset,
        test_dataset,
        subtask=subtask,
        language=language,
        domain=domain,
        model_name_or_path=llm_name,
        seed_run=seed_run,
        num_train_epochs=num_epochs,
        strategy=strategy,
        split_idx=split_idx
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
    parser.add_argument('--split_idx', type=int, default=0,
                        help='Split index for train_split strategy')
    parser.add_argument('--llm_name', type=str,
                        default="unsloth/gemma-3-27b-it-bnb-4bit", help='LLM model name')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs')
    args = parser.parse_args()

    setup_gpu_environment()
    clear_memory()

    subtask = args.subtask
    language = args.language
    domain = args.domain
    seed_run = args.seed_run
    strategy = args.strategy
    split_idx = args.split_idx
    llm_name = args.llm_name
    num_epochs = args.num_epochs
    
    results_path_start = f"results/results_{strategy}/{llm_name.replace('/', '_')}/"
    if strategy == "train_split":
        existing_files = [f for f in os.listdir(results_path_start)
                      if f.startswith(f"{subtask}_{language}_{domain}_0_{split_idx}")]
    else:
        existing_files = [f for f in os.listdir(results_path_start)
                      if f.startswith(f"{subtask}_{language}_{domain}_0")]
    
    if existing_files:
        logger.info(f"Results already exist for subtask={subtask}, language={language}, domain={domain}, seed_run={seed_run}, split_idx={split_idx}. Skipping.")
        return
    

    run_training_pipeline_real(
        subtask, language, domain, seed_run, strategy, split_idx=split_idx, llm_name=llm_name, num_epochs=num_epochs)

    logger.info(f"Experiment completed successfully")
    logger.info(f"="*80)


if __name__ == "__main__":
    main()
