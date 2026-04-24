# CGS-410 COURSE PROJECT

## Human vs LLM Structural Analysis

---

##  About

This repository contains code, datasets, and outputs for analyzing structural patterns in human language and Large Language Models (LLMs).

Focus areas:

* Dependency Length (DL)
* Tree structure (hierarchy)
* Surprisal (prediction difficulty)
* Human vs LLM comparison

---

##  Live Demo

*  **Main Interface:** https://cgs.outputs.muragesh.tech
*  **Outputs & Visualizations:** https://cgs.outputs.muragesh.tech/outputs.html


---

## 📂 Repository Structure

```id="l5y2xf"
CGS-410-COURSE-PROJECT/
│
├── data/
│   ├── AIGTxt_dataset.csv
│   ├── en_ewt-ud-dev.conllu.txt
│   ├── es_ancora-ud-test.conllu
│   ├── fr_gsd-ud-dev.conllu
│   ├── hi_hdtb-ud-dev.conllu
│   ├── mr_ufal-ud-train.conllu
│   ├── de_hdt-ud-dev.conllu
│   ├── llm_generated.txt
│   ├── synthetic_sentences_10000_simple_depths.csv
│   ├── synthetic_sentences_recursive_depths_10000_without_cues.csv
│   └── synthetic_surprisal_dataset_10000.csv
│
├── DataScripts_for_llm_pridiction/
│   ├── generate_data_without_complex.py
│   ├── generate_data_with_inner_complex.py
│   └── generate_data_with_inner_complex_without_removing_cues.py
│
├── Main_analysis_code_files/
│   ├── humanSurprisal.py
│   ├── human_analysis_multi_laug.py
│   ├── human_analysis_single_laug.py
│   ├── llm_analysis_DL_and_TH.py
│   ├── llm_analysis_prediction_data_without_cues.py
│   ├── llm_analysis_prediction_data_without_inner_encoding.py
│   ├── llm_analysis_prediction_data_with_inner_encodng.py
│   ├── llm_vs_human_dl.py
│   ├── multimodel.py
│   ├── human_vs_llm_dl_summary.csv
│   └── multi_model_attention_outputs/
│       ├── attention_depth_summary.csv
│       └── multi_model_surprisal_results.csv
│
├── interactive_llm_visualization_website/
│   └── backend/
│       ├── main.py
│       └── req.txt
│
├── outputs/
│   ├── humanV/
│   │   └── (key result graphs: DL, surprisal, density, etc.)
│   ├── multi_model_llm_surprisal_analysis/
│   │   └── (final multi-model graphs)
│   ├── Outputs_for_human_analysis_multi_laug/
│   ├── Outputs_for_human_analysis_single_laug/
│   ├── Outputs_for_llm_dl_td_Using_AIGTxt_dataset/
│   ├── Outputs_for_LLM_prediction_without_cues/
│   ├── Outputs_for_llm_prediction_wthout_innerembading/
│   ├── Outputs_for_llm_prediction_wth_innerembading/
│   └── (additional experiment outputs & CSV files)
│
├── index.html
├── outputs.html
├── Final_doc_Muragesh.pdf
├── README.md
├── requirements.txt
└── LICENSE
```

# Human vs LLM Structural Analysis  

---

##  Project Overview

This project investigates whether transformer-based Large Language Models (LLMs) exhibit structural patterns similar to human language.  

The focus is on understanding how models handle:
- **Dependency Length (DL)** — structural efficiency of sentences  
- **Hierarchical complexity** — syntactic depth (e.g., center embedding)  
- **Prediction difficulty** — measured using surprisal, perplexity, and entropy  

We compare human language data with LLM-generated text to evaluate whether similar patterns emerge and whether they reflect true cognitive constraints or learned statistical behavior.

---

##  Data Used

### 1. Human Language Data
- Universal Dependencies (UD) datasets  
- Multiple languages (e.g., English, Hindi, Spanish, French, German, Marathi)  
- Provides ground truth syntactic structures  

---

### 2. Synthetic Datasets
- Programmatically generated sentences  
- Controlled variations:
  - Increasing sentence length  
  - Increasing embedding depth  
  - Variants with reduced surface cues
  - around 53000 sentences


---

### 3. LLM Data
- Text generated using transformer models (e.g., GPT-2)  
- Prompts designed to produce varying complexity  
- Also includes AI-generated datasets for large-scale analysis
- ~ 23000 sentences

---

##  Experiments Conducted

### 1. Human Corpus Analysis
- Measured dependency length across languages  
- Analyzed:
  - Average dependency length (ADL)  
  - Distribution of dependencies  
  - Sentence length vs dependency length  
  - Tree height  

---

### 2. LLM Structural Analysis
- Generated text from LLMs  
- Parsed using dependency parsers  
- Compared structural properties with human data  

---

### 3. Dependency Length Comparison
- Human vs LLM comparison  
- Checked whether LLMs follow **Dependency Length Minimization (DLM)**  

---

### 4. Surprisal Analysis
- Measured prediction difficulty using:
  - Token-level surprisal  
  - Sentence perplexity  
  - Entropy  
- Evaluated how difficulty changes with sentence complexity
- compared it for llm vs Human

---

### 5. Hierarchical Complexity Experiment
- Tested increasing syntactic depth (center embedding)  
- Observed effect on:
  - Surprisal  
  - Model prediction behavior  

---

### 6. Surface Cue Ablation
- Removed cues like capitalization patterns  
- Checked whether models rely on:
  - True structure  
  - Or shallow surface patterns  

---

### 7. Multi-Model Analysis
- Compared behavior across multiple transformer models  
- Evaluated consistency of observed patterns  

---

### 8. Attention-Based Analysis
- Analyzed attention weights  
- Checked whether models capture correct long-distance dependencies  

---

##  Goal

To understand whether LLMs:
- Truly model hierarchical structure  
- Exhibit human-like memory constraints  
- Or primarily rely on statistical patterns in data  

---
---

##  How to Run

```bash id="r9l0cb"
pip install -r requirements.txt
```

Run analysis scripts:

```bash id="p0r9kv"
cd Main_analysis_code_files
python <script_name>.py
```

Run backend:

```bash id="o3xj7s"
cd interactive_llm_visualization_website/backend
python main.py
```

---

##  Outputs

All generated graphs, CSV files, and experiment results are stored in:

```id="v1a7hx"
/outputs
```

---

##  Contributors

* Muragesh Nyamagoud
* Palak Meena
* Kovid Saksham Lohia
* Kajal Sankhla

