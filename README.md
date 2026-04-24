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

