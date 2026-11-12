# Farmer Advisory Chatbot Using Multilingual NLP
## A Verification-First Pipeline for Crop Disease Diagnosis and Grounded Treatment Advice

> **Status**: UNDER DEVELOPMENT — results marked "PENDING" have not yet been generated.

---

## Overview

This system provides crop disease diagnosis and treatment advice to farmers in English, Hindi, and Marathi. It combines:

1. **Vision Diagnoser** (trained): A fine-tuned image classifier that identifies crop diseases from leaf photos.
2. **Multilingual Retriever** (trained): A fine-tuned sentence-transformer that retrieves relevant treatment passages from a verified knowledge base.
3. **Critic / Verification Layer** (rule-based): Ensures diagnostic confidence and consistency before providing advice; asks clarifying follow-up questions when uncertain.
4. **Speech & Translation Pipeline** (pretrained): Whisper (STT), NLLB-200 (translation), gTTS (TTS) for multilingual voice interaction.
5. **Gradio Demo**: End-to-end interface accepting leaf photo + text/voice query.

### What This System Does NOT Do
- It does NOT use a generative LLM for advice. All advice text is **extractive** (returned verbatim from the knowledge base with source attribution).
- Agri-R1 (GRPO), EVisRAG/VisRAG, and Vida-Panda are discussed only as **related/future work** in the report. They are NOT implemented in this codebase.

---

## Repository Structure

```
farmer_advisory_chatbot/
├── configs/                    # YAML configuration files for all experiments
│   ├── vision_efficientnet_b0.yaml   # Primary vision model
│   ├── vision_resnet18.yaml          # Ablation: ResNet-18
│   ├── vision_mobilenetv3.yaml       # Ablation: MobileNetV3
│   ├── vision_no_aug.yaml            # Ablation: no augmentation
│   ├── vision_domain_gap.yaml        # Ablation: PV + PD training
│   ├── retriever.yaml                # Retriever fine-tuning
│   ├── critic.yaml                   # Critic thresholds
│   └── demo.yaml                     # Demo app settings
├── data_prep/                  # Data download, preprocessing, and split scripts
│   ├── download_plantvillage.py
│   ├── download_plantdoc.py
│   ├── prepare_splits.py
│   └── generate_retrieval_pairs.py
├── data/                       # Generated splits and templates (gitignored raw data)
│   ├── class_mapping.json
│   ├── test_queries_template.csv
│   └── test_queries_README.md
├── kb/                         # Knowledge base
│   └── passages.json           # Template with TODO_VERIFY fields
├── train_vision.py             # Vision model training script
├── train_retriever.py          # Retriever fine-tuning script
├── critic.py                   # Critic / verification layer
├── eval/                       # Evaluation scripts
│   ├── eval_vision.py
│   ├── eval_retriever.py
│   ├── tune_critic.py
│   └── ablation_summary.py
├── app_gradio.py               # Gradio demo application
├── make_figures.py             # Regenerate all figures from CSV logs
├── notebooks/                  # Google Colab notebooks
│   ├── 01_data.ipynb
│   ├── 02_vision.ipynb
│   ├── 03_retriever.ipynb
│   └── 04_eval_demo.ipynb
├── report_assets/              # Figures, tables, diagrams for the report
├── scripts/                    # Utility scripts
│   └── package.sh
├── requirements.txt
├── .gitignore
└── README.md                   # This file
```

---

## Setup

### Prerequisites
- Python 3.10+ (tested on 3.13.1)
- pip

### Local Installation (CPU)
```bash
cd farmer_advisory_chatbot
pip install -r requirements.txt
```

### Google Colab
Each notebook in `notebooks/` handles its own setup:
1. Mounts Google Drive
2. Installs requirements
3. Downloads data (if needed)
4. Runs training/evaluation
5. Saves outputs to Drive

---

## Quick Start

### 1. Prepare Data
```bash
# Download datasets (see data_prep/ scripts; ~2 GB total)
python data_prep/download_plantvillage.py
python data_prep/download_plantdoc.py
python data_prep/prepare_splits.py
python data_prep/generate_retrieval_pairs.py
```

### 2. Train Vision Model (CPU smoke test)
```bash
python train_vision.py --config configs/vision_efficientnet_b0.yaml --smoke
```

### 3. Train Retriever (CPU smoke test)
```bash
python train_retriever.py --config configs/retriever.yaml --smoke
```

### 4. Run Evaluation
```bash
python eval/eval_vision.py --config configs/vision_efficientnet_b0.yaml
python eval/eval_retriever.py --config configs/retriever.yaml
python eval/tune_critic.py --config configs/critic.yaml
```

### 5. Launch Demo
```bash
python app_gradio.py --config configs/demo.yaml
```

### 6. Generate Report Figures
```bash
python make_figures.py
```

---

## Experiments & Ablations

| # | Experiment | Config(s) | Key Metrics |
|---|-----------|-----------|-------------|
| 1 | Vision backbone comparison | `vision_efficientnet_b0.yaml`, `vision_resnet18.yaml`, `vision_mobilenetv3.yaml` | Accuracy, Macro-F1, #Params, CPU latency |
| 2 | Domain gap | `vision_efficientnet_b0.yaml`, `vision_domain_gap.yaml` | PlantDoc test accuracy |
| 3 | Augmentation ablation | `vision_efficientnet_b0.yaml`, `vision_no_aug.yaml` | Train/Val accuracy |
| 4 | Retrieval: zero-shot vs fine-tuned | `retriever.yaml` | Recall@1, Recall@5, MRR per language |
| 5 | Language strategy | `retriever.yaml` | Metrics + latency (direct vs translate-pivot) |
| 6 | Critic thresholds | `critic.yaml` | Accuracy on answered, coverage, abstention rate |

**Results**: PENDING (run Colab notebooks to generate)

---

## Expected Run Times (Colab T4)

| Task | Approx. Time |
|------|-------------|
| Vision training (20 epochs, ~15k images) | ~25-35 min |
| Retriever training (5 epochs, ~30k pairs) | ~10-15 min |
| Full evaluation suite | ~10 min |
| Demo startup | ~2 min (model loading) |

---

## Regenerating Figures and Tables

All figures and tables are generated from CSV log files. After training:

```bash
python make_figures.py
```

This produces:
- `report_assets/vision_loss_curve.png` / `.pdf`
- `report_assets/retriever_loss_curve.png` / `.pdf`
- `report_assets/confusion_matrix.png` / `.pdf`
- `report_assets/selective_prediction_curve.png` / `.pdf`
- `report_assets/ablation_table.csv` / `.tex`
- `report_assets/methodology_diagram.png` / `.svg`

---

## Packaging for Submission

```bash
bash scripts/package.sh
```
Creates `GroupXX_code_Farmer_Advisory_Chatbot.zip` excluding datasets and large checkpoints.

---

## Citation & References

> **Note**: The following are related works referenced in the report. Verify all citations on Google Scholar before including in the final report.

- "PlantVillage: A Public Dataset for Plant Disease Detection" — VERIFY ON GOOGLE SCHOLAR
- "PlantDoc: A Dataset for Visual Plant Disease Detection" — VERIFY ON GOOGLE SCHOLAR
- "Multilingual E5 Text Embeddings" — VERIFY ON GOOGLE SCHOLAR
- "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" — VERIFY ON GOOGLE SCHOLAR

---

## License

This project is an academic submission. See individual dataset licenses:
- PlantVillage: CC0 (public domain)
- PlantDoc: Verify license from source

---

## Team

- Group: GroupXX (replace with actual group number)
- Course: Semester 7 Project

