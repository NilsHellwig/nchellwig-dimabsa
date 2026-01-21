# Self-Consistent Structured Generation (SCSG) for Dimensional Aspect-Based Sentiment Analysis

<div align="center">

**SemEval-2026 Task 3 · Track A Submission · Subtask 2 & 3**

[![Paper](https://img.shields.io/badge/Paper-SemEval%202026-blue?style=for-the-badge&logo=googlescholar)](tba)
[![Model](https://img.shields.io/badge/Gemma--3--27B-FFAB00?style=for-the-badge&logo=googlegemini)](https://huggingface.co/google/gemma-3-27b-it)

---

**Nils Constantin Hellwig¹ · Jakob Fehle¹ · Udo Kruschwitz² · Christian Wolff¹**

¹Media Informatics Group · ²Information Science Group  
University of Regensburg, Germany

---

*Instruction-tuned LLMs with self-consistency for consistent Dimensional Aspect-Based Sentiment Analysis*

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
- 20GB+ VRAM recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/NilsHellwig/nchellwig-dimabsa
cd nchellwig-dimabsa

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

| Parameter      | Description         | Options                                  |
| -------------- | ------------------- | ---------------------------------------- |
| `--subtask`    | Task variant        | `2` (DimASTE), `3` (DimASQP)             |
| `--language`   | Target language     | `eng`, `zho`, `jpn`, `rus`, `tat`, `ukr` |
| `--domain`     | Review domain       | `restaurant`, `laptop`, `hotel`          |
| `--llm_name`   | Base model          | `unsloth/gemma-3-27b-it-bnb-4bit`, etc.  |
| `--strategy`   | Evaluation strategy | `test-train_dev`, `dev-train`            |
| `--num_epochs` | Training epochs     | Integer (default: 5)                     |

## 📊 Results

### Evaluation Metrics

- **Continuous Precision, Recall and F1 Score** for Subtasks 2 & 3
- **Precision & Recall** metrics
- **Cross-lingual performance** analysis

Results are organized in the `exported_predictions/` directory with detailed breakdowns by:

- Language (English, Chinese, Japanese, Russian, Tatar, Ukrainian)
- Domain (Restaurant, Laptop, Hotel)

## 🌍 Supported Languages & Domains

| Language  | Code  | Domains            |
| --------- | ----- | ------------------ |
| English   | `eng` | Restaurant, Laptop |
| Chinese   | `zho` | Restaurant, Laptop |
| Japanese  | `jpn` | Hotel              |
| Russian   | `rus` | Restaurant         |
| Tatar     | `tat` | Restaurant         |
| Ukrainian | `ukr` | Restaurant         |

## 🔗 Resources

- 📄 [SemEval-2026 Task 3 Overview](tba)
- 📘 [Task Repository](https://github.com/DimABSA/DimABSA2026)
- 💾 [Dataset](https://github.com/DimABSA/DimABSA2026/tree/main/task-dataset)
- 🤖 [Gemma-3 Model (Unsloth)](https://huggingface.co/unsloth/gemma-3-27b-it-bnb-4bit)

## 📬 Contact

<div align="center">

| Author                      | Affiliation               | Email                                                                 |
| --------------------------- | ------------------------- | --------------------------------------------------------------------- |
| **Nils Constantin Hellwig** | Media Informatics Group   | [nils-constantin.hellwig@ur.de](mailto:nils-constantin.hellwig@ur.de) |
| **Jakob Fehle**             | Media Informatics Group   | [jakob.fehle@ur.de](mailto:jakob.fehle@ur.de)                         |
| **Udo Kruschwitz**          | Information Science Group | [udo.kruschwitz@ur.de](mailto:udo.kruschwitz@ur.de)                   |
| **Christian Wolff**         | Media Informatics Group   | [christian.wolff@ur.de](mailto:christian.wolff@ur.de)                 |

**University of Regensburg, Germany**

🐙 GitHub: [@NilsHellwig](https://github.com/NilsHellwig)

</div>

## 📖 Citation

```bibtex
tba
```

<div align="center">
Made with ❤️ at the Faculty of Informatics and Data Science, University of Regensburg
</div>
