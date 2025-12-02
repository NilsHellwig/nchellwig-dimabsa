#!/usr/bin/env python3
import sys
import subprocess

# Valid combinations of (subtask, language, domain) that have data
VALID_COMBINATIONS = [
    (2, "eng", "restaurant"),
    (2, "eng", "laptop"),
    (2, "jpn", "hotel"),
    (2, "rus", "restaurant"),
    (2, "tat", "restaurant"),
    (2, "ukr", "restaurant"),
    (2, "zho", "restaurant"),
    (2, "zho", "laptop"),
    (3, "eng", "restaurant"),
    (3, "eng", "laptop"),
    (3, "jpn", "hotel"),
    (3, "rus", "restaurant"),
    (3, "tat", "restaurant"),
    (3, "ukr", "restaurant"),
    (3, "zho", "restaurant"),
    (3, "zho", "laptop"),
]

N_SEEDS_RUNS = 1
STRATEGY = "train_split"  # "evaluation" oder "train_split"
N_SPLITS = 5  # Anzahl der 80/20 Splits für train_split
EPOCHS = [5]
LLMs = ["unsloth/gemma-3-27b-it-bnb-4bit"]

for num_epochs in [5]:
    for seed_run in range(N_SEEDS_RUNS):
        for llm in LLMs:
            for subtask, language, domain in VALID_COMBINATIONS:
                for split_idx in range(N_SPLITS):
                    cmd = [
                        sys.executable,
                        "train_llm.py",
                        "--subtask", str(subtask),
                        "--language", language,
                        "--domain", domain,
                        "--seed_run", str(seed_run),
                        "--strategy", STRATEGY,
                        "--llm_name", llm,
                        "--num_epochs", str(num_epochs),
                        "--split_idx", str(split_idx)
                    ]
                    subprocess.run(cmd)
