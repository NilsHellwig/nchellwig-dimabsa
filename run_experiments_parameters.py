#!/usr/bin/env python3
import sys
import subprocess

SUBTASKS = [3, 2]
LANGUAGES = ["rus", "eng", "jpn", "tat", "ukr", "zho"]
DOMAINS = ["restaurant", "laptop", "hotel", "finance"]
N_SEEDS_RUNS = 1
STRATEGY = "train_split"  # "pred_dev" oder "train_split"
N_SPLITS = 5  # Anzahl der 80/20 Splits für train_split
EPOCHS = [5]
LLMs = ["unsloth/gemma-3-4b-it-bnb-4bit"]

# STUDIE mit cross-validation:
# 1a. Prediction mit temp=0 ohne guided decoding
# 1b. Prediction mit temp=0 mit guided decoding

# 2a. Prediction mit temp=0.8 -> 15 mal gleiche prompt ausführen ohne guided decoding
# 2b. Prediction mit temp=0.8 -> 15 mal gleiche prompt ausführen mit guided decoding


for num_epochs in [5]:
    for seed_run in range(N_SEEDS_RUNS):
        for llm in LLMs:
            for subtask in SUBTASKS:
                for domain in DOMAINS:
                    for split_idx in range(N_SPLITS):
                        for language in LANGUAGES:
                            cmd = [
                                sys.executable,
                                "train_llm.py",
                                "--subtask", str(subtask),
                                "--language", language,
                                "--domain", domain,
                                "--seed_run", str(seed_run),
                                "--strategy", "pred_dev",
                                "--llm_name", llm,
                                "--num_epochs", str(num_epochs),
                                "--split_idx", str(split_idx)
                            ]
                            subprocess.run(cmd)
