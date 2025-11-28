#!/usr/bin/env python3
import sys
import subprocess

SUBTASKS = [2, 3]
LANGUAGES = ["rus", "eng", "jpn", "tat", "ukr", "zho"]
DOMAINS = ["restaurant", "laptop", "hotel", "finance"]
N_SEEDS_RUNS = 1
N_EPOCHS = 5

# STUDIE mit cross-validation:
# 1a. Prediction mit temp=0 ohne guided decoding
# 1b. Prediction mit temp=0 mit guided decoding

# 2a. Prediction mit temp=0.8 -> 15 mal gleiche prompt ausführen ohne guided decoding
# 2b. Prediction mit temp=0.8 -> 15 mal gleiche prompt ausführen mit guided decoding


for seed_run in range(N_SEEDS_RUNS):
    for subtask in SUBTASKS:
        for domain in DOMAINS:
            for language in LANGUAGES:
                cmd = [
                    sys.executable,
                    "train_llm.py",
                    "--subtask", str(subtask),
                    "--language", language,
                    "--domain", domain,
                    "--seed_run", str(seed_run),
                    "--strategy", "pred_dev",
                    "--llm_name", "unsloth/gemma-3-27b-it-bnb-4bit",
                    "--num_epochs", str(N_EPOCHS)
                ]
                subprocess.run(cmd)
