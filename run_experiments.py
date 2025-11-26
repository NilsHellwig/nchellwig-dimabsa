#!/usr/bin/env python3
import sys
import subprocess

SUBTASKS = [3, 2]
LANGUAGES = ["eng", "jpn", "rus", "tat", "ukr", "zho"]
DOMAINS = ["restaurant", "laptop", "hotel", "finance"]
N_SEEDS_RUNS = 1
STRATEGY = "train_split"  # "pred_dev" oder "train_split"
N_SPLITS = 5  # Anzahl der 80/20 Splits für train_split
EPOCHS = [5, 10, 15]

for seed_run in range(N_SEEDS_RUNS):
    for llm in ["unsloth/gemma-3-4b-it-bnb-4bit", "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"]:
        for num_epochs in EPOCHS:
            for language in LANGUAGES:
                for subtask in SUBTASKS:
                    for domain in DOMAINS:
                        for split_idx in range(N_SPLITS):
                            cmd = [
                                sys.executable,
                                "train_llm.py",
                                "--subtask", str(subtask),
                                "--language", language,
                                "--domain", domain,
                                "--seed_run", str(seed_run),
                                "--strategy", "train_split",
                                "--split_idx", str(split_idx),
                                "--llm_name", llm,
                                "--num_epochs", str(num_epochs)
                            ]
                            subprocess.run(cmd)
