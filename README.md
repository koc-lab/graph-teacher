<!-- markdownlint-disable MD013 -->

# GraphTeacher

The source for the _GraphTeacher: Transductive Fine-Tuning of Encoders through Graph Neural Networks_ paper, published in IEEE TAI.

GraphTeacher introduces a semi-supervised fine-tuning framework that augments Transformer encoders with Graph Neural Networks (GNNs). The method leverages unlabeled samples without accessing test nodes during graph construction (no leakage) and does not require re-graphing at inference, enabling scalable inductive deployment.

> This repository contains the official implementation accompanying the paper.


---

## 🔑 Key Features

- Semi-supervised fine-tuning for Transformer models
- Graph built only over labeled + unlabeled **training** data
- **No test-node usage** → avoids information leakage
- **Inductive inference** → new samples do not require re-graphing
- Works with BERT, DistilBERT, RoBERTa
- Significant gains in low-label settings (5–50% labeled data)

---

## 📂 Repository Structure

| Path | Description |
|------|------------|
| `graph_teacher/` | Core GraphTeacher implementation |
| `baseline/` | Baseline Transformer fine-tuning code |
| `commons/` | Shared utilities (config, dataset loaders, helpers) |
| **`sweep_graph_teacher.py`** | ✅ **Main script** — runs GraphTeacher sweeps |
| `sweep_baseline.py` | Sweep for baseline models |
| `exp_book_gnn.py` | GraphTeacher experimental runner |
| `exp_book_baseline.py` | Baseline experimental runner |
| `pyproject.toml` | Project dependency configuration |
| `uv.lock` | Locked environment file |
| `.gitignore` | Ignore rules |
| `LICENSE` | License file |
| `README.md` | This file |

---

## 🚀 Quick Start



---

## 📦 Requirements

- Python ≥ 3.9
- PyTorch
- HuggingFace Transformers
- DGL or PyTorch Geometric
- Weights & Biases (`wandb`)
- NumPy, pandas, tqdm, matplotlib

> Please install all required libraries before running the scripts.



