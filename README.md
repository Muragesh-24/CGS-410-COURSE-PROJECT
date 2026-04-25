# CGS-410 COURSE PROJECT

*  **Web Tool:** https://cgs.outputs.muragesh.tech
*  **Outputs & Visualizations:** https://cgs.outputs.muragesh.tech/outputs.html

## Human vs LLM Structural Analysis


## 📁 Project Structure

```
CGS-410-COURSE-PROJECT/
│
├── data/
│   ├── Human_dependency_length/
│   │   ├── de_hdt-ud-dev.conllu
│   │   ├── en_ewt-ud-dev.conllu.txt
│   │   ├── es_ancora-ud-test.conllu
│   │   ├── fr_gsd-ud-dev.conllu
│   │   ├── hi_hdtb-ud-dev.conllu
│   │   └── mr_ufal-ud-train.conllu
│   │
│   ├── LLM_Dependency_Length/
│   │   ├── AIGTxt_dataset.csv
│   │   ├── french.csv
│   │   ├── hindi.csv
│   │   ├── kannadda.csv
│   │   └── spanish.csv
│   │
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
│   ├── llm_analysis_multi_laug_DL.py
│   ├── llm_analysis_prediction_data_without_cues.py
│   ├── llm_analysis_prediction_data_without_inner_encoding.py
│   ├── llm_analysis_prediction_data_with_inner_encodng.py
│   ├── llm_vs_human_dl.py
│   └── multimodel.py
│
├── interactive_llm_visualization_website/
│   └── backend/
│       ├── main.py
│       └── req.txt
│
├── outputs/
│   ├── humanV/
│   │   ├── density.png
│   │   ├── DL.png
│   │   ├── human_vs_llm_dl_summary.csv
│   │   ├── human_vs_llm_surprisal.png
│   │   ├── max_surprisal.png
│   │   └── sLLm.png
│   │
│   ├── llm_analysis_more_parameters/
│   ├── LLM_ANALYSIS_ONLY_FIRSTLETTER_CAPS_small_dataset/
│   ├── llm_dl_td_small_dataset/
│   ├── llm_multilingual_dl_simple/
│   ├── multi_model_attention_outputs/
│   ├── multi_model_llm_surprisal_analysis/
│   ├── Outputs_for_human_analysis_multi_laug/
│   ├── Outputs_for_human_analysis_single_laug/
│   ├── Outputs_for_llm_dl_td_Using_AIGTxt_dataset/
│   ├── Outputs_for_LLM_prediction_without_cues/
│   ├── Outputs_for_llm_prediction_wthout_innerembading/
│   ├── Outputs_for_llm_prediction_wth_innerembading/
│   └── synthetic_sentences_with_prediction.csv
│
├── index.html
├── outputs.html
├── muragesh_240669_finalD0c_cgs.pdf
├── README.md
├── requirements.txt
├── LICENSE
└── CNAME
```
---

##  About
## Data Used

| Category | Dataset / File | Language | Quantity (sentences) | Purpose |
|---|---|---:|---:|---|
| Human corpus | `Human_dependency_length/de_hdt-ud-dev.conllu` | German | ~18,434 | Additional human corpus |
| Human corpus | `Human_dependency_length/en_ewt-ud-dev.conllu.txt` | English | ~2,002 | Human dependency-length baseline |
| Human corpus | `Human_dependency_length/es_ancora-ud-test.conllu` | Spanish | ~1,721 | Human dependency-length baseline |
| Human corpus | `Human_dependency_length/hi_hdtb-ud-dev.conllu` | Hindi | ~1,659 | Human dependency-length baseline |
| Human corpus | `Human_dependency_length/fr_gsd-ud-dev.conllu` | French | ~1,476 | Human dependency-length baseline |
| Human corpus | `Human_dependency_length/mr_ufal-ud-train.conllu` | Marathi | ~373 | Human dependency-length baseline |
| **Total (Human)** |  |  | **~25,665** |  |
| LLM corpus | `LLM_Dependency_Length/AIGTxt_dataset.csv` | English | ~23,000 | LLM-generated dependency-length analysis |
| LLM corpus | `LLM_Dependency_Length/french.csv` | French | 1,616 | LLM-generated multilingual comparison |
| LLM corpus | `LLM_Dependency_Length/hindi.csv` | Hindi | 1,243 | LLM-generated multilingual comparison |
| LLM corpus | `LLM_Dependency_Length/spanish.csv` | Spanish | 802 | LLM-generated multilingual comparison |
| LLM corpus | `LLM_Dependency_Length/kannadda.csv` | Kannada | 280 | LLM-generated multilingual comparison |
| LLM text | `llm_generated.txt` | English | 661 | LLM-generated text sample |
| **Total (LLM)** |  |  | **~27,602** |  |
| Synthetic | `synthetic_sentences_recursive_depths_10000_without_cues.csv` | English | 10,000 | Recursive-depth experiment without surface cues |
| Synthetic | `synthetic_surprisal_dataset_10000.csv` | English | 10,000 | Surprisal and hierarchy experiment |
| Synthetic | `synthetic_sentences_10000_simple_depths.csv` | English | 8,580 | Controlled depth / sentence-length experiment |
| **Total (Synthetic)** |  |  | **28,580** |  |
| **Grand Total** |  |  | **~81,847 sentences** |  |


Focus areas:

* Dependency Length (DL)
* Tree structure (hierarchy)
* Surprisal (prediction difficulty)
* Human vs LLM comparison

---

##  Live Web tool & Outputs
###  This is a web tool built by us from the scratch to visualize the working of llm and also a collection of all the outputs and visualizations that we have generated as part of our project(ie outputs of our code).

1) **Web Tool:** This is an interactive web tool that display the working of llm using a gpt 2 model , it show how a llm tokenizes a sentence, how it generates the next token and how the surprisal changes with each token. It also shows the attention weights of the model and how it captures long distance dependencies. You can try it out with your own sentences and see how the model responds.
*  **Web Tool:** https://cgs.outputs.muragesh.tech

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
<img src="outputs/Web_tool_demo/Screenshot 2026-04-25 044027.png" alt="Website Screenshot" width="400">
<img src="outputs/Web_tool_demo/Screenshot 2026-04-25 044046.png" alt="Website Screenshot" width="400">
<img src="outputs/Web_tool_demo/Screenshot 2026-04-25 044100.png" alt="Website Screenshot" width="400">
<img src="outputs/Web_tool_demo/Screenshot 2026-04-25 044128.png" alt="Website Screenshot" width="400">
</div>

2) **Outputs & Visualizations:** This is a collection of all the outputs and visualizations that we have generated as part of our project. It includes graphs and other results from our experiments. You can explore them in detail and see how they support our analysis and conclusions.
*  **Outputs & Visualizations:** https://cgs.outputs.muragesh.tech/outputs.html
---


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

