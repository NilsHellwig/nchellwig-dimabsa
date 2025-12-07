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
N_EPOCHS = 5
LLMs = [ "unsloth/gemma-3-27b-it-bnb-4bit", "unsloth/gemma-3-12b-it-bnb-4bit"]

for llm in LLMs:
  for seed_run in range(N_SEEDS_RUNS):
    for subtask, language, domain in VALID_COMBINATIONS:
        cmd = [
                sys.executable,
            "train_llm.py",
            "--subtask", str(subtask),
            "--language", language,
            "--domain", domain,
            "--seed_run", str(seed_run),
            "--strategy", "evaluation",
            "--llm_name", llm,
            "--num_epochs", str(N_EPOCHS)
        ]
        subprocess.run(cmd)
