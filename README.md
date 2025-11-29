# nchellwig at SemEval-2026 Task 3: Self-Consistent Structured Generation (SCSG) for Dimensional Aspect-Based Sentiment Analysis using Large Language Models (Code)

**Author:** Nils Hellwig ([@NilsHellwig](https://github.com/NilsHellwig)), University of Regensburg  
**Task:** Track A - Dimensional Aspect-Based Sentiment Analysis (DimABSA)  
**Subtasks:** 2 (DimASTE) & 3 (DimASQP)  

## 📋 Overview

This repository contains our submission for the SemEval 2026 DimABSA shared task. We tackle Subtasks 2 and 3 of Track A, focusing on extracting dimensional sentiment triplets and quadruplets from text using fine-tuned Large Language Models (LLMs).

Our approach leverages the Gemma-3 model family with LoRA fine-tuning, incorporating guided decoding for structured output generation and temperature-based sampling for robustness evaluation.

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU

### Installation
```bash
pip install -r evaluation_script/requirements.txt
```

## 🎯 Usage

### Running Experiments
- **Parameter Sweeps:** `python run_experiments_parameters.py`
- **Final Evaluation:** `python run_final.py`

### Training a Model
```bash
python train_llm.py --subtask 3 --language eng --domain restaurant --llm_name unsloth/gemma-3-27b-it-bnb-4bit --num_epochs 5
```

### Key Parameters
- `--subtask`: 2 or 3
- `--language`: eng, zho, etc.
- `--domain`: restaurant, laptop, etc.
- `--strategy`: pred_dev or train_split
- `--llm_name`: Model identifier

## 📊 Results

Results are stored in `results/` directory with different configurations:
- Temperature 0.0 (deterministic)
- Temperature 0.8 (stochastic, multiple runs)
- With/without guided decoding

Evaluation metrics include continuous F1 for Subtasks 2&3.

## 📁 Structure

- `llm.py`: Core training and evaluation logic
- `train_llm.py`: Main training script
- `run_*.py`: Experiment runners
- `helper.py`: Utility functions
- `evaluation_script/`: Official evaluation tools
- `task-dataset/`: Training data
- `results/`: Output predictions

## 🔗 Links

- [SemEval 2026 DimABSA](https://www.aclweb.org/portal/content/call-participation-semeval-2026-task-3-dimensional-aspect-based-sentiment-analysis-customer)
- [Task Description](https://github.com/DimABSA/DimABSA2026)
- [Dataset](https://github.com/DimABSA/DimABSA2026/tree/main/task-dataset)
- [Contact: Nils-Constantin.Hellwig@ur.de](mailto:Nils-Constantin.Hellwig@ur.de)

## Citation
If you use this code for your research, please cite our work as follows:
```bibtex
...to be added...
```

---
