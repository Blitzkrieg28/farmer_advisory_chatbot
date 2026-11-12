# Farmer Advisory Chatbot Using Multilingual NLP
## A Verification-First Pipeline for Crop Disease Diagnosis and Grounded Treatment Advice

> **Status**: **VERIFIED & REPRODUCIBLE** — End-to-end pipeline validated across Vision Diagnosis, In-The-Wild Domain Gap Ablations, Dense Cross-Lingual Retrieval, Rule-Based Critic Verification, and Gradio Web Demo.

---

## 1. Overview & System Philosophy

Smallholder farmers across India face severe crop losses due to plant diseases, yet agricultural extension advice is frequently delayed, linguistically inaccessible, or dangerously hallucinated by generic AI tools. 

This repository implements a **verification-first, strictly extractive agricultural advisory system** supporting English, Hindi, and Marathi. The system links:

1. **Vision Diagnoser (Trained)**: Fine-tuned CNN classifier (`EfficientNet-B0`) identifying crop diseases from leaf macro-photographs across 14 disease/healthy classes. Evaluated under both laboratory (PlantVillage) and real unconstrained field conditions (PlantDoc).
2. **Dense Multilingual Retriever (Trained)**: Domain-adapted sentence-transformer (`intfloat/multilingual-e5-small`) fine-tuned with Multiple Negatives Ranking Loss (MNRL) to retrieve grounded treatment passages from 429 verified chunks of official ICAR & State Agricultural University extension manuals.
3. **Critic / Verification Layer (Rule-Based)**: A deterministic 3-way decision engine that eliminates hallucinations:
   - **ACCEPT** ($p_1 \ge \tau \land \text{margin} \ge \delta$): Delivers verbatim extractive advice with exact document, page number, and chemical dosage flags.
   - **CLARIFY** ($\text{margin} < \delta$): Detects confusable disease pairs (e.g., Tomato Early Blight vs. Late Blight) and prompts the farmer with distinguishing physical leaf symptoms (Option A vs Option B).
   - **REFUSE** ($p_1 < \tau_{refuse}$ or Out-of-Distribution): Safely abstains and refers the farmer to the nearest Krishi Vigyan Kendra (KVK) center.
4. **Speech & Translation Integration**: Speech-to-text voice input via Whisper, cross-lingual querying, and text-to-speech advisory pronunciation via gTTS in English, Hindi, and Marathi.
5. **Interactive Gradio Web Demo**: Responsive 3-pane interface (Left: Leaf upload & vision probabilities; Center: Multilingual Chat & interactive Option A/B clarification; Right: Verified advisory card with verbatim citation, dosage warning, and audio playback).

### Strict Academic Constraints
- **Zero Generative LLM Hallucinations**: No generative language model writes chemical names or dosages. All treatment text is extracted **verbatim** from official extension PDFs.
- **Academic Scoping**: Frontier speculative techniques (VisRAG, GRPO-based Agri-R1, and multi-agent debate) are analyzed only as related/future work in the final project report.

---

## 2. Experimental Benchmark Results (100% Real GPU Data)

### A. Multilingual Dense Retrieval Benchmark (N = 711 Evaluation Queries)

Benchmarked on 711 balanced test queries (237 English, 237 Hindi, 237 Marathi) evaluated against the 429-chunk knowledge base. 95% Confidence Intervals computed via 500-iteration BCa bootstrap. Latencies measured on an NVIDIA T4 GPU.

| Configuration | Language | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) | MRR@10 [95% CI] | Latency (ms) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **BM25 Direct** (Sparse Baseline) | Overall | 7.7% | 18.6% | 24.5% | 0.125 [0.104, 0.145] | 0.9 ms |
| | English | 20.3% | 46.4% | 58.2% | 0.313 [0.267, 0.360] | 1.3 ms |
| | Hindi | 1.7% | 5.1% | 8.0% | 0.034 [0.016, 0.055] | 0.7 ms |
| | Marathi | 1.3% | 4.2% | 7.2% | 0.028 [0.012, 0.046] | 0.9 ms |
| **multilingual-e5 Zero-Shot** | Overall | 18.6% | 42.8% | 53.6% | 0.296 [0.269, 0.322] | 18.5 ms |
| | English | 29.5% | 64.6% | 74.3% | 0.448 [0.398, 0.500] | 15.8 ms |
| | Hindi | 16.0% | 40.5% | 51.9% | 0.267 [0.221, 0.314] | 11.8 ms |
| | Marathi | 10.1% | 23.2% | 34.6% | 0.172 [0.134, 0.214] | 11.9 ms |
| **multilingual-e5 Fine-Tuned (Ours)** | **Overall** | **33.6%** | **68.8%** | **79.3%** | **0.486 [0.456, 0.514]** | **19.2 ms** |
| | **English** | **42.2%** | **81.4%** | **89.5%** | **0.588 [0.541, 0.635]** | **12.0 ms** |
| | **Hindi** | **31.7%** | **64.6%** | **76.8%** | **0.458 [0.406, 0.507]** | **14.7 ms** |
| | **Marathi** | **27.0%** | **60.3%** | **71.7%** | **0.413 [0.363, 0.463]** | **14.4 ms** |
| **Fine-Tuned + English Pivot** | Overall | 34.2% | 69.5% | 80.1% | 0.492 [0.462, 0.521] | 21.0 ms |

> **Key Takeaway**: Sparse BM25 completely fails on Indic scripts (5.1% R@5 in Hindi, 4.2% in Marathi) due to morphological inflections and vocabulary mismatch with English extension PDFs. Fine-tuning `multilingual-e5-small` with MNRL yields a **+60.8% relative gain** in overall R@5 (68.8% vs 42.8%) and a **+270% relative gain** over BM25, bringing Hindi R@5 to 64.6% and Marathi to 60.3%.

---

### B. Vision Diagnoser & In-The-Wild Domain Gap Ablation (N = 3,128 Lab vs. N = 127 Wild Field)

To evaluate real-world deployability, we benchmarked the vision classifier on both the canonical PlantVillage test set (in-vitro lab conditions) and the PlantDoc test set (in-the-wild farm conditions):

| Model Backbone | Training Data | Epochs | PlantVillage In-Vitro Acc | PlantVillage Macro-F1 | PlantDoc In-The-Wild Acc | PlantDoc Wild Macro-F1 | CPU Latency | Parameters |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| EfficientNet-B0 | PlantVillage Only (Lab) | 20 | 99.90% | 99.88% | 30.71% | 21.83% | 35.12 ± 3.6 ms | 4,025,482 |
| **EfficientNet-B0** | **PV + PlantDoc (Adapted)** | **20** | **99.71%** | **99.66%** | **66.14%** (+115.4% rel.) | **63.44%** | **36.48 ± 3.5 ms** | **4,025,482** |

> **Key Conclusions & Insights**:
> 1. **The Laboratory Illusion**: Training purely on laboratory images (PlantVillage) yields an almost flawless 99.90% lab accuracy, but suffers severe **domain collapse (dropping to 30.71%)** on natural farm photos due to complex backgrounds, multi-leaf occlusions, and uncontrolled sunlight.
> 2. **Field Adaptation Gain**: Joint training on in-the-wild PlantDoc samples more than doubles real-world field accuracy from **30.71% to 66.14%** and nearly triples Macro-F1 from **21.83% to 63.44%**, while maintaining 99.71% accuracy on clean images.
> 3. **Need for Critic Verification**: Because real field conditions cause confidence degradation, the rule-based Critic layer is physically indispensable: it intercepts ambiguous low-margin predictions and disambiguates them through physical symptom questions.

---

### C. Critic Verification Layer: Selective Risk vs. Coverage Calibration

Evaluated across 54 threshold combinations on 500 validation calibration samples (`results/critic_tuning_results.csv`):

| Operating Point | Thresholds ($\tau, \delta$) | Direct Coverage (%) | Effective Coverage (%) | Clarification Rate (%) | Selective Risk (%) |
|---|:---:|:---:|:---:|:---:|:---:|
| Permissive Baseline | $\tau=0.50, \delta=0.05$ | 74.4% | 85.2% | 10.8% | 12.90% |
| Intermediate | $\tau=0.60, \delta=0.10$ | 61.0% | 85.2% | 24.2% | 7.54% |
| **Calibrated Optimal** | **$\tau=0.70, \delta=0.15$** | **58.6%** | **85.2%** | **26.6%** | **6.48%** |

> **Key Takeaway**: Enforcing margin threshold $\delta = 0.15$ cuts selective risk (rate of providing incorrect advice) **in half** (from 12.9% down to 6.48%). For near-tie cases, asking the farmer a 2-choice distinguishing physical symptom question restores effective coverage back to **85.2%** while safely filtering out out-of-distribution leaf inputs.

---

## 3. How to Run the Web Demo

The demo provides an interactive 3-pane Gradio interface with multilingual text/speech input and verified advisory cards.

### Method 1: Local Desktop / Laptop Launch
```bash
# 1. Activate your virtual environment and install dependencies
pip install -r requirements.txt

# 2. Launch the Gradio web server
python app_gradio.py --config configs/demo.yaml
```
- Open your browser and navigate to: **`http://localhost:7860`**
- By default, if GPU weights are present in `checkpoints/`, it runs on GPU; otherwise, it automatically utilizes CPU mode.

### Method 2: Google Colab Public Launch (Shareable Link)
To run in Colab and generate a public link accessible from any smartphone or browser:
```bash
!python app_gradio.py --config configs/demo.yaml --share
```
- Gradio will display a public tunnel URL (e.g. `https://xxxx.gradio.live`) valid for 72 hours.

### Method 3: Headless CLI Smoke Test (No Browser Needed)
To verify the entire demo pipeline on CPU without opening a browser:
```bash
python app_gradio.py --smoke
```
This tests all 3 decision branches:
- **Test 1 (ACCEPT)**: Clear English query $\rightarrow$ returns diagnosis + verified advice card.
- **Test 2 (CLARIFY)**: Ambiguous Hindi query $\rightarrow$ presents physical symptom disambiguation question.
- **Test 2b (RESOLVE)**: Simulates selecting Option A $\rightarrow$ resolves diagnosis and returns grounded advice.
- **Test 3 (REFUSE)**: Blurry / OOD Marathi query $\rightarrow$ returns safe abstention with Marathi KVK referral.

---

### Step-by-Step Demo User Guide

1. **Select Language**: Choose between `English`, `हिंदी (Hindi)`, or `मराठी (Marathi)` in the top dropdown.
2. **Upload Leaf Photo (Left Pane)**:
   - Upload any tomato, potato, corn, or grape leaf image.
   - The left pane displays the model's top-2 diagnostic predictions and confidence margin.
3. **Ask a Treatment Question (Center Pane)**:
   - **Text**: Type your query (e.g., *"What fungicide should I spray and what is the dosage?"* or *"पत्तियों पर धब्बे दिख रहे हैं, कौन सी दवा छिड़कें?"*).
   - **Voice**: Click the microphone icon to record your spoken question (transcribed by Whisper STT).
4. **Interactive Clarification**:
   - If the system detects a near-tie between two confusable diseases (e.g., Tomato Early Blight vs. Late Blight), the Critic will display **Option A** and **Option B** buttons describing distinctive physical symptoms.
   - Click either option to resolve the ambiguity.
5. **Inspect Verified Advisory Card (Right Pane)**:
   - Displays **verbatim text** directly from the official ICAR / State Agricultural University PDF.
   - Shows the exact **Document Title**, **Section**, and **Page Number**.
   - Displays dosage alerts (e.g., *"Mancozeb 75 WP @ 2.5 g/L"*) and Pre-Harvest Interval (PHI) warnings.
   - Click the audio player to listen to natural **spoken audio pronunciation** in your selected language via gTTS.

---

## 4. Repository Structure

```
farmer_advisory_chatbot/
├── configs/                          # Experiment configs & mappings
│   ├── class_mapping.json            # 14 unified crop disease classes
│   ├── critic.yaml                   # Calibrated thresholds & physical symptom bank
│   ├── demo.yaml                     # Web demo & speech TTS settings
│   ├── disease_synonyms.yaml         # Multilingual agricultural terms (EN, HI, MR)
│   ├── retriever.yaml                # Retriever training & evaluation settings
│   └── vision_efficientnet_b0.yaml   # Primary EfficientNet-B0 vision model
├── data/                             # Generated splits & test benchmarks (gitignored raw data)
│   ├── class_mapping.json
│   ├── retrieval_train.csv           # 4,149 training pairs (generated via Qwen + NLLB)
│   ├── retrieval_val.csv             # 711 evaluation queries (EN, HI, MR)
│   └── test_queries_template.csv     # 50 verified human-annotated test queries
├── data_prep/                        # Preprocessing & synthetic pair generation
│   ├── generate_queries.py           # Domain query generation with leakage guard
│   └── ingest_pdfs.py                # Verbatim PDF chunking & dosage tagger
├── eval/                             # Evaluation & calibration scripts
│   ├── eval_retriever.py             # Cross-lingual retrieval benchmark & bootstrap CIs
│   ├── eval_vision.py                # Vision classifier evaluation & latency profiler
│   └── tune_critic.py                # Grid search for optimal tau and delta thresholds
├── kb/                               # Grounded knowledge base
│   ├── chunks.jsonl                  # 429 verified PDF chunks with exact citations
│   └── pdfs/                         # Official ICAR / SAU extension manuals
├── logs/                             # Training loss & metric CSVs
│   ├── retriever/e5_small_finetuned_seed42_metrics.csv
│   ├── vision/efficientnet_b0_plantvillage_seed42_metrics.csv
│   └── vision/efficientnet_b0_pv_pd_seed42_metrics.csv
├── notebooks/                        # Self-contained Google Colab notebooks
│   ├── 01_data.ipynb                 # Ingestion & pair generation
│   ├── 02_vision.ipynb               # Vision training on Colab T4 GPU
│   ├── 03_retriever.ipynb            # Retriever fine-tuning with MNRL
│   └── 04_eval_demo.ipynb            # Full benchmark evaluation & Gradio share
├── report_assets/                    # 300 DPI publication figures & LaTeX tables
│   ├── ablation_table.csv/.tex       # Multilingual retrieval ablation tables
│   ├── vision_ablation_table.csv/.tex# Vision & domain gap ablation tables
│   ├── domain_gap_ablation.png/.pdf  # PlantVillage vs. PlantDoc domain gap bar chart
│   ├── confusion_matrix.png/.pdf     # Real in-field confusion matrix heatmaps
│   ├── methodology_diagram.png/.svg  # End-to-end pipeline architecture diagram
│   ├── retrieval_comparison_bar.png/.pdf # R@5 & MRR@10 comparison bar chart
│   ├── retriever_loss_curve.png/.pdf # Retriever MNRL loss & MRR curves
│   ├── selective_prediction_curve.png/.pdf # Critic coverage vs. risk tradeoff
│   └── vision_loss_curve.png/.pdf    # Real 20-epoch vision CE loss & accuracy curves
├── results/                          # Benchmark results CSV files
│   ├── critic_tuning_results.csv     # 54 threshold calibration grid results
│   ├── retrieval_results.csv         # 711-query retrieval benchmark metrics
│   ├── efficientnet_b0_plantvillage_seed42_preds_plantdoc_test.csv
│   └── efficientnet_b0_pv_pd_seed42_preds_plantdoc_test.csv
├── scripts/                          # Utility & submission packaging scripts
│   ├── package.py                    # Cross-platform submission packager
│   ├── package.sh                    # Linux/macOS submission script
│   └── package.ps1                   # Windows PowerShell packaging script
├── app_gradio.py                     # Interactive 3-pane Gradio Web Demo
├── critic.py                         # Deterministic Critic verification layer
├── make_figures.py                   # Script regenerating all report figures & tables
├── pipeline.py                       # Unified FarmerAdvisoryPipeline module
├── retrieval.py                      # BM25 + dense sentence-transformer retriever
├── train_retriever.py                # Retriever fine-tuning script
├── train_vision.py                   # Vision classifier training script
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 5. Regenerating Report Figures & Tables

All report assets are programmatically reproducible from the underlying experiment CSV logs:

```bash
python make_figures.py
```

This outputs 300 DPI figures in both raster (PNG) and vector (PDF / SVG) formats under `report_assets/`:
- `report_assets/methodology_diagram.{png,svg}`: Pipeline architecture flowchart.
- `report_assets/vision_loss_curve.{png,pdf}`: Real 20-epoch training/validation loss and accuracy curves.
- `report_assets/retriever_loss_curve.{png,pdf}`: Real 5-epoch MNRL training loss and validation MRR@10 curves.
- `report_assets/domain_gap_ablation.{png,pdf}`: PlantVillage vs. PlantDoc laboratory vs. wild field accuracy.
- `report_assets/confusion_matrix.{png,pdf}`: Real in-field confusion heatmaps comparing PV vs. PV+PD models.
- `report_assets/retrieval_comparison_bar.{png,pdf}`: R@5 and MRR@10 across languages with 95% bootstrap CIs.
- `report_assets/selective_prediction_curve.{png,pdf}`: Selective risk vs. coverage tradeoff curve.
- `report_assets/ablation_table.{csv,tex}`: Complete LaTeX `booktabs` and CSV retrieval benchmark tables.
- `report_assets/vision_ablation_table.{csv,tex}`: Complete LaTeX `booktabs` and CSV vision domain gap tables.

---

## 6. Packaging for Academic Submission

To build the submission zip archive (excluding large raw images, checkpoints, `.git`, and cache files):

**Linux / macOS:**
```bash
bash scripts/package.sh Group01 Farmer_Advisory_Chatbot_Multilingual_NLP
```

**Windows PowerShell:**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/package.ps1 Group01 Farmer_Advisory_Chatbot_Multilingual_NLP
```

Outputs: `Group01_code_Farmer_Advisory_Chatbot_Multilingual_NLP.zip` (~37 MB).

---

## 7. Academic References

1. **PlantVillage Dataset**:
   Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science, 7, 1419.
2. **PlantDoc In-Field Dataset**:
   Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection in Natural Field Conditions.* Proceedings of the 7th ACM IKDD CoDS and 25th COMAD, 249–253.
3. **Multilingual E5 Embeddings**:
   Wang, L., Yang, N., Huang, F., Jiao, B., Yang, L., Jiang, D., Majumder, R., & Wei, F. (2022). *Text Embeddings by Weakly-Supervised Contrastive Pre-training.* arXiv:2212.03533.
4. **EfficientNet Architecture**:
   Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019, 6105–6114.
5. **Whisper Speech Recognition**:
   Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision.* ICML 2023.
6. **NLLB Multilingual Translation**:
   NLLB Team. (2022). *No Language Left Behind: Scaling Human-Centered Machine Translation.* arXiv:2207.04672.
