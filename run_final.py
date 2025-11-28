#!/usr/bin/env python3
import sys
import subprocess

SUBTASKS = [3, 2]
LANGUAGES = ["rus", "eng", "jpn", "tat", "ukr", "zho"]
DOMAINS = ["restaurant", "laptop", "hotel", "finance"]
N_SEEDS_RUNS = 1
STRATEGY = "pred_dev"  # oder "pred_test"
EPOCHS = [5]

### STUDIE mit cross-validation: 
# 1a. Prediction mit temp=0 ohne guided decoding
# 1b. Prediction mit temp=0 mit guided decoding

# 2a. Prediction mit temp=0.8 -> 15 mal gleiche prompt ausführen ohne guided decoding
# 2b. Prediction mit temp=0.8 -> 15 mal gleiche prompt ausführen mit guided decoding


for num_epochs in [5]:
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
                                "--num_epochs", str(num_epochs)
                            ]
                            subprocess.run(cmd)
