# 🎯 ur-mi-nch - SemEval 2026 Task 3

This repository contains the **ur-mi-nch** submission for **SemEval 2026 Task 3: Dimensional Aspect-Based Sentiment Analysis (DimABSA)** - Subtasks 2 and 3.

## 📖 Overview

Traditional Aspect-Based Sentiment Analysis (ABSA) uses coarse-grained categorical sentiment labels (positive, negative, neutral). This task bridges the gap between ABSA and dimensional sentiment analysis by representing sentiment along **continuous valence-arousal (VA) dimensions**, inspired by established theories in psychology and affective science (Russell, 1980; 2003).

### Valence-Arousal Representation

- **Valence**: Measures the degree of positivity or negativity (1.00 = extremely negative, 9.00 = extremely positive, 5.00 = neutral)
- **Arousal**: Measures the intensity of emotion (1.00 = very low arousal, 9.00 = very high arousal, 5.00 = medium)

**Example Comparison:**
> "The salads are fantastic."  
> → `salads`, `FOOD#QUALITY`, `fantastic`, `Positive`

**Dimensional ABSA:**
> "The salads are fantastic."  
> → `salads`, `FOOD#QUALITY`, `fantastic`, `7.88#7.75`

## 🎯 Implemented Subtasks

### Subtask 2: Dimensional Aspect Sentiment Triplet Extraction (DimASTE)

Given a text, extract all **(A, O, VA)** triplets:
- **A**: Aspect term (e.g., "salads", "battery")
- **O**: Opinion term (e.g., "fantastic", "terrible")
- **VA**: Valence-Arousal score (e.g., "7.88#7.75")

**Input Format (JSON Lines):**
```json
{"ID": "123", "Text": "The salads are fantastic."}
```

**Output Format (JSON Lines):**
```json
{"ID": "123", "Triplet": [{"Aspect": "salads", "Opinion": "fantastic", "VA": "7.88#7.75"}]}
```

### Subtask 3: Dimensional Aspect Sentiment Quadruplet Extraction (DimASQE)

Given a text, extract all **(A, C, O, VA)** quadruplets:
- **A**: Aspect term
- **C**: Aspect category (Entity#Attribute format, e.g., "FOOD#QUALITY")
- **O**: Opinion term
- **VA**: Valence-Arousal score

**Input Format (JSON Lines):**
```json
{"ID": "123", "Text": "The salads are fantastic."}
```

**Output Format (JSON Lines):**
```json
{"ID": "123", "Quadruplet": [{"Aspect": "salads", "Category": "FOOD#QUALITY", "Opinion": "fantastic", "VA": "7.88#7.75"}]}
```

## 🏗️ Architecture

### Model
- **Base Model**: Gemma-3 27B Instruct (4-bit quantized via Unsloth)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
  - Rank: 64
  - Alpha: 16
  - Target modules: Attention and MLP layers

### Training Pipeline
1. **Data Loading**: Loads training/dev datasets from `task-dataset/track_a/`
2. **Prompt Generation**: Multilingual prompts tailored to each language
3. **Fine-tuning**: Supervised fine-tuning with chat template formatting
4. **Inference**: vLLM for efficient batch inference with LoRA adapters
5. **Output Parsing**: Extracts triplets/quadruplets from model outputs

## 📁 Project Structure

```
.
├── train_llm.py              # Main training script
├── llm.py                    # Training and evaluation logic
├── helper.py                 # Utility functions (prompts, parsing, etc.)
├── run_experiments.bash      # Batch experiment runner
├── logs.txt                  # Training logs with timestamps
├── task-dataset/             # Dataset files
│   └── track_a/
│       ├── subtask_2/
│       └── subtask_3/
├── results/
    └── results_pred_dev/     # Predictions for dev set
        ├── subtask_2/
        └── subtask_3/

```

## 🚀 Usage

### Single Experiment

```bash
python train_llm.py \
    --subtask 3 \
    --language eng \
    --domain restaurant \
    --seed_run 0 \
    --strategy pred_dev
```

### Batch Experiments

Run all experiments across languages, domains, and subtasks:

```bash
bash run_experiments.bash
```

This will automatically:
- Skip non-existent language-domain combinations
- Save predictions in the correct format
- Log progress to `logs.txt`

### Parameters

- `--subtask`: Task number (2 or 3)
- `--language`: Language code (eng, jpn, rus, tat, ukr, zho)
- `--domain`: Domain name (restaurant, laptop, hotel, finance)
- `--seed_run`: Random seed for reproducibility
- `--strategy`: 
  - `pred_dev`: Train on train split, evaluate on dev split (default)
  - `train_split`: Split train data for internal evaluation

## 📊 Output Format

### For `pred_dev` Strategy

Predictions are saved in submission-ready format:
```
results/results_pred_dev/subtask_2/pred_eng_restaurant.jsonl
results/results_pred_dev/subtask_3/pred_zho_laptop.jsonl
```

### For Other Strategies

```
results/predictions_{subtask}_{language}_{domain}_{seed}_{strategy}.jsonl
```

---

*Built with 🔥 using Unsloth and vLLM for efficient LLM fine-tuning*
