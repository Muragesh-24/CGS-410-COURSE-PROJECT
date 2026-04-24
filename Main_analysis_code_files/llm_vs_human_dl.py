import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from conllu import parse
from scipy.stats import linregress

# ==============================
# CONFIG
# ==============================

LLM_CSV_PATH = "../data/AIGTxt_dataset.csv"
LLM_TEXT_COLUMN = "ChatGPT-Generated"

HUMAN_FILES = {
    "English": "../data/en_ewt-ud-dev.conllu.txt",
    "French": "../data/fr_gsd-ud-dev.conllu",
    "Hindi": "../data/hi_hdtb-ud-dev.conllu",
    "Marathi": "../data/mr_ufal-ud-train.conllu",
    "Spanish": "../data/es_ancora-ud-test.conllu",
}

nlp = spacy.load("en_core_web_sm")


# ==============================
# HUMAN DEPENDENCY ANALYSIS
# ==============================

def analyze_human_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = parse(f.read())

    sent_lengths = []
    avg_dls = []
    all_dls = []

    for sentence in sentences:
        total_dl = 0
        dep_count = 0

        valid_tokens = [t for t in sentence if type(t["id"]) == int]
        sent_len = len(valid_tokens)

        for token in valid_tokens:
            head = token["head"]

            if head != 0:
                dl = abs(token["id"] - head)
                total_dl += dl
                dep_count += 1
                all_dls.append(dl)

        if dep_count > 0:
            sent_lengths.append(sent_len)
            avg_dls.append(total_dl / dep_count)

    return sent_lengths, avg_dls, all_dls


human_lengths = []
human_avg_dls = []
human_all_dls = []

for lang, path in HUMAN_FILES.items():
    print(f"Processing human data: {lang}")
    x, y, dist = analyze_human_file(path)

    human_lengths.extend(x)
    human_avg_dls.extend(y)
    human_all_dls.extend(dist)


# ==============================
# LLM DEPENDENCY ANALYSIS
# ==============================

def dependency_metrics_spacy(doc):
    total_dl = 0
    dep_count = 0
    all_dls = []
    

    for token in doc:
        if token.head != token:
            dl = abs(token.i - token.head.i)
            total_dl += dl
            dep_count += 1
            all_dls.append(dl)

    if dep_count == 0:
        return None, []

    return total_dl / dep_count, all_dls


df = pd.read_csv(LLM_CSV_PATH, encoding="latin1")
llm_sentences = df[LLM_TEXT_COLUMN].dropna().astype(str).tolist()

llm_lengths = []
llm_avg_dls = []
llm_all_dls = []

print("\nProcessing LLM data...")

for i, sentence in enumerate(llm_sentences):
    if i % 1000 == 0:
        print("Processed:", i)

    doc = nlp(sentence)

    avg_dl, dist = dependency_metrics_spacy(doc)

    if avg_dl is not None:
        llm_lengths.append(len(doc))
        llm_avg_dls.append(avg_dl)
        llm_all_dls.extend(dist)


# ==============================
# SUMMARY + GROWTH RATE
# ==============================

def summarize(name, lengths, avg_dls, all_dls):
    slope, intercept, r, p, stderr = linregress(lengths, avg_dls)

    return {
        "Source": name,
        "Sentence Count": len(lengths),
        "Mean Sentence Length": np.mean(lengths),
        "Mean Avg DL": np.mean(avg_dls),
        "Median Avg DL": np.median(avg_dls),
        "Max DL": np.max(all_dls),
        "DL Growth Rate": slope,
        "Correlation": r
    }


human_summary = summarize("Human", human_lengths, human_avg_dls, human_all_dls)
llm_summary = summarize("LLM", llm_lengths, llm_avg_dls, llm_all_dls)

summary_df = pd.DataFrame([human_summary, llm_summary])

print("\n========== HUMAN VS LLM DEPENDENCY LENGTH COMPARISON ==========")
print(summary_df.round(3))

summary_df.to_csv("human_vs_llm_dl_summary.csv", index=False)


# ==============================
# BINNED TREND FUNCTION
# ==============================

def binned_average(lengths, values, bin_size=5, max_len=60):
    bins = np.arange(0, max_len + bin_size, bin_size)

    bin_centers = []
    bin_means = []

    for i in range(len(bins) - 1):
        low = bins[i]
        high = bins[i + 1]

        selected = [
            values[j]
            for j in range(len(lengths))
            if low < lengths[j] <= high
        ]

        if len(selected) > 0:
            bin_centers.append((low + high) / 2)
            bin_means.append(np.mean(selected))

    return bin_centers, bin_means


human_x, human_y = binned_average(human_lengths, human_avg_dls)
llm_x, llm_y = binned_average(llm_lengths, llm_avg_dls)


# ==============================
# PLOT 1: GROWTH COMPARISON
# ==============================

plt.figure(figsize=(7, 5))

plt.plot(human_x, human_y, marker="o", linewidth=2, label="Human Text")
plt.plot(llm_x, llm_y, marker="o", linewidth=2, label="LLM Text")

plt.xlabel("Sentence Length")
plt.ylabel("Average Dependency Length")
plt.title("Human vs LLM: Dependency Length Growth")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("human_vs_llm_dl_growth.png", dpi=300)
plt.show()


# ==============================
# PLOT 2: DISTRIBUTION COMPARISON
# ==============================

plt.figure(figsize=(7, 5))

plt.hist(human_all_dls, bins=40, alpha=0.6, density=True, label="Human Text")
plt.hist(llm_all_dls, bins=40, alpha=0.6, density=True, label="LLM Text")

plt.xlabel("Dependency Length")
plt.ylabel("Density")
plt.title("Human vs LLM: Dependency Length Distribution")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("human_vs_llm_dl_distribution.png", dpi=300)
plt.show()


print("\nSaved files:")
print("1. human_vs_llm_dl_growth.png")
print("2. human_vs_llm_dl_distribution.png")
print("3. human_vs_llm_dl_summary.csv")