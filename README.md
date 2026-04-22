
## 📁 Project Structure

```text
CGS-410-COURSE-PROJECT/
│
├── data/
│ ├── AIGTxt_dataset.csv
│ ├── de_hdt-ud-dev.conllu
│ ├── en_ewt-ud-dev.conllu.txt
│ ├── es_ancora-ud-test.conllu
│ ├── fr_gsd-ud-dev.conllu
│ ├── hi_hdtb-ud-dev.conllu
│ ├── mr_ufal-ud-train.conllu
│ ├── llm_generated.txt
│ ├── synthetic_sentences_10000_simple_depths.csv
│ ├── synthetic_sentences_recursive_depths_10000_without_cues.csv
│ └── synthetic_surprisal_dataset_10000.csv
│
├── DataScripts_for_llm_pridiction/
│ ├── generate_data_without_complex.py
│ ├── generate_data_with_inner_complex.py
│ └── generate_data_with_inner_complex_without_removing_cues.py
│
├── Main_analysis_code_files/
│ ├── human_analysis_multi_laug.py
│ ├── human_analysis_single_laug.py
│ ├── llm_analysis_DL_and_TH.py
│ ├── llm_analysis_prediction_data_without_cues.py
│ ├── llm_analysis_prediction_data_without_inner_encoding.py
│ └── llm_analysis_prediction_data_with_inner_encodng.py
│
├── interactive_llm_visualization_website/
│ └── backend/
│ ├── main.py
│ └── req.txt
│
├── outputs/
│ ├── llm_analysis_more_parameters/
│ ├── LLM_ANALYSIS_ONLY_FIRSTLETTER_CAPS_small_dataset/
│ ├── llm_dl_td_small_dataset/
│ ├── Outputs_for_human_analysis_multi_laug/
│ ├── Outputs_for_human_analysis_single_laug/
│ ├── Outputs_for_llm_dl_td_Using_AIGTxt_dataset/
│ ├── Outputs_for_LLM_prediction_without_cues/
│ ├── Outputs_for_llm_prediction_wthout_innerembading/
│ ├── Outputs_for_llm_prediction_wth_innerembading/
│ └── synthetic_sentences_with_prediction.csv
│
├── index.html
├── outputs.html
├── requirements.txt
├── LICENSE
├── CNAME
└── README.md
```

---

##  Directory Overview

###  `data/`
Contains all datasets used in the project:
- Human language corpora (Universal Dependencies)
- LLM-generated text samples
- Synthetic datasets for controlled experiments (depth, embedding, surprisal)

 Acts as the **primary input source**

---

###  `DataScripts_for_llm_pridiction/`
Scripts used to generate controlled datasets:
- Create sentences with increasing syntactic complexity
- Generate variants with/without structural cues

 Used for **controlled linguistic experiments**

---

###  `Main_analysis_code_files/`
Core research and analysis logic:
- Human dependency length and tree structure analysis
- LLM structural analysis (DL, tree height)
- Surprisal and prediction difficulty experiments
- Correlation and statistical analysis

 Represents the **main experimental pipeline**

---

###  `interactive_llm_visualization_website/`
Interactive system for real-time analysis:
- Flask backend for:
  - Token-level surprisal
  - Prediction probabilities
  - Attention visualization

 Enables **interactive exploration of model behavior**

---

###  `outputs/`
All experiment results and visualizations:
- Graphs (DL, surprisal trends, comparisons)
- CSV result files
- Structured outputs for each experiment

 Serves as the **final results layer**

---

###  Root Files
- `index.html` → Interactive webpage
- `outputs.html` → Visualization dashboard
- `requirements.txt` → Python dependencies
- `README.md` → Documentation
- `CNAME` → Custom domain configuration

---

##  Workflow Overview


<h2>Collaborators</h2>

<ul>
<li>Muragesh Channappa Nyamagoud</li>
<li>Palak Meena</li>
<li>Kovid Saksham Lohia</li>
<li>Kajal Sankhla</li>
</ul>
