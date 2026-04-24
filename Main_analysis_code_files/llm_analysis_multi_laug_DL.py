import os
import pandas as pd
import spacy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

OUTPUT_DIR = "outputs/llm_multilingual_dl_simple"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "English": {
        "path": "data/LLM_Dependency_Length/AIGTxt_dataset.csv",
        "type": "csv",
        "column": "ChatGPT-Generated",
        "model": "en_core_web_sm"
    },
    "French": {
        "path": "data/LLM_Dependency_Length/french.csv",
        "type": "txt",
        "column": None,
        "model": "fr_core_news_sm"
    },
    "Hindi": {
        "path": "data/LLM_Dependency_Length/hindi.csv",
        "type": "txt",
        "column": None,
        "model": "en_core_web_sm"   # approximate only
    },
    "spanish": {
        "path": "data/LLM_Dependency_Length/spanish.csv",
        "type": "txt",
        "column": None,
        "model": "es_core_news_sm"   
    }
}

LIMIT = None   # set 1000 for testing, None for all


def read_sentences(config):
    path = config["path"]

    if config["type"] == "csv":
        df = pd.read_csv(path, encoding="latin1")
        sentences = df[config["column"]].dropna().astype(str).tolist()
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sentences = [line.strip() for line in f if line.strip()]

    if LIMIT:
        sentences = sentences[:LIMIT]

    return sentences


def dependency_metrics(doc):
    total_dl = 0
    dep_count = 0
    max_dep = 0

    for token in doc:
        if token.head != token:
            dl = abs(token.i - token.head.i)
            total_dl += dl
            dep_count += 1
            max_dep = max(max_dep, dl)

    if dep_count == 0:
        return 0, 0

    return total_dl / dep_count, max_dep


def get_tree_height(token):
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(get_tree_height(child) for child in children)


def sentence_tree_height(doc):
    root = None
    for token in doc:
        if token.head == token:
            root = token
            break

    if root is None:
        return 0

    return get_tree_height(root)


all_summary = []
all_sentence_rows = []

for language, config in FILES.items():

    print(f"\n========== Processing {language} ==========")

    sentences = read_sentences(config)
    print("Total sentences:", len(sentences))

    nlp = spacy.load(config["model"])

    avg_dl = []
    max_dl = []
    tree_heights = []
    sent_lengths = []

    for i, doc in enumerate(nlp.pipe(sentences, batch_size=64)):

        if i % 500 == 0:
            print(f"{language}: processed {i}/{len(sentences)}")

        avg, maxdep = dependency_metrics(doc)
        height = sentence_tree_height(doc)

        avg_dl.append(avg)
        max_dl.append(maxdep)
        tree_heights.append(height)
        sent_lengths.append(len(doc))

        all_sentence_rows.append({
            "language": language,
            "sentence_id": i + 1,
            "sentence": doc.text,
            "sentence_length": len(doc),
            "avg_dependency_length": avg,
            "max_dependency_length": maxdep,
            "tree_height": height
        })

    print("\nSentence statistics")
    print("Average sentence length:", np.mean(sent_lengths))
    print("Maximum sentence length:", np.max(sent_lengths))

    print("\nDependency statistics")
    print("Average dependency length:", np.mean(avg_dl))
    print("Maximum dependency length:", np.max(max_dl))

    print("\nTree statistics")
    print("Average tree height:", np.mean(tree_heights))
    print("Maximum tree height:", np.max(tree_heights))

    if len(sent_lengths) > 1:
        corr_len_dl, _ = pearsonr(sent_lengths, avg_dl)
        corr_height_dl, _ = pearsonr(tree_heights, avg_dl)
    else:
        corr_len_dl = 0
        corr_height_dl = 0

    print("\nCorrelation analysis")
    print("Sentence Length vs Avg DL:", corr_len_dl)
    print("Tree Height vs Avg DL:", corr_height_dl)

    all_summary.append({
        "language": language,
        "total_sentences": len(sentences),
        "avg_sentence_length": np.mean(sent_lengths),
        "max_sentence_length": np.max(sent_lengths),
        "avg_dependency_length": np.mean(avg_dl),
        "max_dependency_length": np.max(max_dl),
        "avg_tree_height": np.mean(tree_heights),
        "max_tree_height": np.max(tree_heights),
        "corr_sentence_length_avg_dl": corr_len_dl,
        "corr_tree_height_avg_dl": corr_height_dl
    })

    plt.figure(figsize=(6, 5))
    plt.hist(avg_dl, bins=30)
    plt.xlabel("Average Dependency Length")
    plt.ylabel("Frequency")
    plt.title(f"{language} LLM Dependency Length Distribution")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{language}_dl_distribution.png", dpi=300)
    plt.show()


summary_df = pd.DataFrame(all_summary)
sentence_df = pd.DataFrame(all_sentence_rows)

summary_df.to_csv(f"{OUTPUT_DIR}/llm_multilingual_dl_summary.csv", index=False)
sentence_df.to_csv(f"{OUTPUT_DIR}/llm_multilingual_sentence_level.csv", index=False)

# -----------------------------
# FINAL COMBINED GRAPHS ONLY
# -----------------------------

summary_df = pd.DataFrame(all_summary)
sentence_df = pd.DataFrame(all_sentence_rows)

summary_df.to_csv(f"{OUTPUT_DIR}/llm_multilingual_dl_summary.csv", index=False)
sentence_df.to_csv(f"{OUTPUT_DIR}/llm_multilingual_sentence_level.csv", index=False)

print("\n========== FINAL SUMMARY ==========")
print(summary_df)


# Graph 1: Average DL across languages
plt.figure(figsize=(8, 5))
plt.bar(summary_df["language"], summary_df["avg_dependency_length"])
plt.xlabel("Language")
plt.ylabel("Average Dependency Length")
plt.title("Average Dependency Length Across LLM-Generated Languages")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/combined_avg_dl_languages.png", dpi=300)
plt.show()


# Graph 2: Combined DL distribution
plt.figure(figsize=(9, 6))

for language in sentence_df["language"].unique():
    temp = sentence_df[sentence_df["language"] == language]
    plt.hist(
        temp["avg_dependency_length"],
        bins=30,
        alpha=0.45,
        density=True,
        label=language
    )

plt.xlabel("Average Dependency Length")
plt.ylabel("Density")
plt.title("Dependency Length Distribution Across Languages")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/combined_dl_distribution_languages.png", dpi=300)
plt.show()


# Graph 3: Sentence length vs Average DL
plt.figure(figsize=(8, 6))

for language in sentence_df["language"].unique():
    temp = sentence_df[sentence_df["language"] == language]
    plt.scatter(
        temp["sentence_length"],
        temp["avg_dependency_length"],
        alpha=0.45,
        label=language
    )

plt.xlabel("Sentence Length")
plt.ylabel("Average Dependency Length")
plt.title("Sentence Length vs Dependency Length Across Languages")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/combined_sentence_length_vs_dl.png", dpi=300)
plt.show()


# Graph 4: Tree height vs Average DL
plt.figure(figsize=(8, 6))

for language in sentence_df["language"].unique():
    temp = sentence_df[sentence_df["language"] == language]
    plt.scatter(
        temp["tree_height"],
        temp["avg_dependency_length"],
        alpha=0.45,
        label=language
    )

plt.xlabel("Tree Height")
plt.ylabel("Average Dependency Length")
plt.title("Tree Height vs Dependency Length Across Languages")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/combined_tree_height_vs_dl.png", dpi=300)
plt.show()


# Graph 5: Average Tree Height across languages
plt.figure(figsize=(8, 5))
plt.bar(summary_df["language"], summary_df["avg_tree_height"])
plt.xlabel("Language")
plt.ylabel("Average Tree Height")
plt.title("Average Tree Height Across LLM-Generated Languages")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/combined_avg_tree_height_languages.png", dpi=300)
plt.show()