# Self-Consistent Structured Generation for Dimensional Aspect-Based Sentiment Analysis

<div align="center">

**SemEval-2026 Task 3 - Track A Submission**

[![Paper](https://img.shields.io/badge/Paper-SemEval%202026-blue)](https://www.aclweb.org/portal/content/call-participation-semeval-2026-task-3-dimensional-aspect-based-sentiment-analysis-customer)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Gemma--3-orange.svg)](https://huggingface.co/google/gemma-3-27b-it)

*Fine-tuned LLMs with guided decoding for structured sentiment extraction*

**Nils Constantin Hellwig¹ · Jakob Fehle¹ · Udo Kruschwitz² · Christian Wolff¹**

¹Media Informatics Group · ²Information Science Group  
University of Regensburg, Germany

</div>

---

## 🎯 Overview

This repository contains our submission for the **SemEval 2026 DimABSA** (Dimensional Aspect-Based Sentiment Analysis) shared task. We tackle **Subtasks 2 and 3** of Track A, focusing on extracting dimensional sentiment triplets (DimASTE) and quadruplets (DimASQP) from customer reviews using fine-tuned Large Language Models with a self-consistent structured generation approach.

Our approach leverages the **Gemma-3 model family** with LoRA fine-tuning, incorporating:

- 🔄 **Self-Consistency** mechanisms for improved predictions
- 🌍 **Multi-lingual Support** across 8 languages and 3 domains

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (for training)
- 20GB+ RAM recommended

### Setup
```bash
# Clone the repository
git clone https://github.com/NilsHellwig/ur-mi-nch.git
cd ur-mi-nch

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Training a Model

```bash
# Train a model for Subtask 3 (DimASQP) on English restaurant reviews
python train_llm.py \
  --subtask 3 \
  --language eng \
  --domain restaurant \
  --llm_name unsloth/gemma-3-27b-it-bnb-4bit \
  --num_epochs 5
```

#### Key Parameters
| Parameter | Description | Options |
|-----------|-------------|---------|
| `--subtask` | Task variant | `2` (DimASTE), `3` (DimASQP) |
| `--language` | Target language | `eng`, `zho`, `jpn`, `rus`, `tat`, `ukr` |
| `--domain` | Review domain | `restaurant`, `laptop`, `hotel` |
| `--llm_name` | Base model | `unsloth/gemma-3-27b-it-bnb-4bit`, etc. |
| `--strategy` | Evaluation strategy | `pred_dev`, `train_split` |
| `--num_epochs` | Training epochs | Integer (default: 5) |

### Running Experiments

```bash
# Parameter sweep across configurations
python run_experiments_parameters.py

# Final evaluation with optimal parameters
python run_final.py
```

## 📊 Results

### Evaluation Metrics
- **Continuous F1 Score** for Subtasks 2 & 3
- **Precision & Recall** metrics
- **Cross-lingual performance** analysis

Results are organized in the `results/` directory with detailed breakdowns by:
- Language (English, Chinese, Japanese, Russian, Tatar, Ukrainian)
- Domain (Restaurant, Laptop, Hotel)
- Configuration (temperature, guidance, sampling strategy)

## 🌍 Supported Languages & Domains

| Language | Code | Domains |
|----------|------|---------|
| English | `eng` | Restaurant, Laptop |
| Chinese | `zho` | Restaurant, Laptop |
| Japanese | `jpn` | Hotel |
| Russian | `rus` | Restaurant |
| Tatar | `tat` | Restaurant |
| Ukrainian | `ukr` | Restaurant |

## 🔗 Resources

- 📄 [SemEval-2026 Task 3 Overview](https://www.aclweb.org/portal/content/call-participation-semeval-2026-task-3-dimensional-aspect-based-sentiment-analysis-customer
- 📘 [Task Repository](https://github.com/DimABSA/DimABSA2026)
- 💾 [Dataset](https://github.com/DimABSA/DimABSA2026/tree/main/task-dataset)
- 🤖 [Gemma-3 Model](https://huggingface.co/google/gemma-3-27b-it)

## 📬 Contact

<div align="center">

| Author | Affiliation | Email |
|--------|-------------|-------|
| **Nils Constantin Hellwig** | Media Informatics Group | [nils-constantin.hellwig@ur.de](mailto:nils-constantin.hellwig@ur.de) |
| **Jakob Fehle** | Media Informatics Group | [jakob.fehle@ur.de](mailto:jakob.fehle@ur.de) |
| **Udo Kruschwitz** | Information Science Group | [udo.kruschwitz@ur.de](mailto:udo.kruschwitz@ur.de) |
| **Christian Wolff** | Media Informatics Group | [christian.wolff@ur.de](mailto:christian.wolff@ur.de) |

**University of Regensburg, Germany**

🐙 GitHub: [@NilsHellwig](https://github.com/NilsHellwig)

</div>

## 📖 Citation


```bibtex
tba
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ at the University of Regensburg
</div>
