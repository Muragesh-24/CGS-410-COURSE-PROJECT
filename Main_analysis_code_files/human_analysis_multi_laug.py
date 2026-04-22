"""
MULTI-LANGUAGE DEPENDENCY ANALYSIS (FINAL VERSION)

Includes:
- Dependency Length stats
- Sentence Length vs DL
- Tree Height
- Cross-language comparison
- Advanced visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
from conllu import parse
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns


# =============================================================================
# FUNCTIONS
# =============================================================================

def analyze_dependency_lengths(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = parse(f.read())

    sentence_avg = []
    all_lengths = []
    max_dl = 0

    for sentence in sentences:
        total = 0
        deps = 0

        for token in sentence:
            if type(token["id"]) != int:
                continue

            if token["head"] != 0:
                dl = abs(token["id"] - token["head"])
                all_lengths.append(dl)
                total += dl
                deps += 1
                max_dl = max(max_dl, dl)

        if deps > 0:
            sentence_avg.append(total / deps)

    overall = sum(all_lengths) / len(all_lengths)

    return {
        "overall_avg": overall,
        "max_dl": max_dl,
        "distribution": all_lengths
    }


def sentence_length_vs_dl(file_path):
    with open(file_path, encoding="utf-8") as f:
        sentences = parse(f.read())

    sentence_lengths = []
    sentence_dl = []

    for sentence in sentences:
        total = 0
        deps = 0

        for token in sentence:
            if type(token["id"]) != int:
                continue

            if token["head"] != 0:
                total += abs(token["id"] - token["head"])
                deps += 1

        if deps > 0:
            sentence_lengths.append(len(sentence))
            sentence_dl.append(total / deps)

    return sentence_lengths, sentence_dl


def tree_height(sentence):
    heads = {}

    for token in sentence:
        if type(token["id"]) == int:
            heads[token["id"]] = token["head"]

    def get_depth(node):
        depth = 0
        while node != 0:
            node = heads.get(node, 0)
            depth += 1
        return depth

    return max(get_depth(node) for node in heads)


# =============================================================================
# DATA PATHS (UPDATE IF NEEDED)
# =============================================================================

languages = {
    "English": "../data/en_ewt-ud-dev.conllu.txt",
    "French": "../data/fr_gsd-ud-dev.conllu",
    "Hindi": "../data/hi_hdtb-ud-dev.conllu",
    "Marathi": "../data/mr_ufal-ud-train.conllu",
    "Spanish": "../data/es_ancora-ud-test.conllu",
}

results = {}

print("\n" + "="*70)
print("MULTI-LANGUAGE ANALYSIS")
print("="*70)


# =============================================================================
# MAIN LOOP
# =============================================================================

for lang, path in languages.items():
    print(f"\n🔍 Processing: {lang}")

    dl_data = analyze_dependency_lengths(path)
    x, y = sentence_length_vs_dl(path)
    corr, _ = pearsonr(x, y)

    with open(path, "r", encoding="utf-8") as f:
        sentences = parse(f.read())

    heights = [tree_height(s) for s in sentences]

    results[lang] = {
        "avg_dl": dl_data["overall_avg"],
        "max_dl": dl_data["max_dl"],
        "std_dl": np.std(dl_data["distribution"]),
        "median_dl": np.median(dl_data["distribution"]),
        "p90_dl": np.percentile(dl_data["distribution"], 90),
        "corr": corr,
        "avg_height": np.mean(heights),
        "max_height": max(heights),
        "distribution": dl_data["distribution"]
    }

    print(f"Avg DL: {results[lang]['avg_dl']:.3f}")
    print(f"Max DL: {results[lang]['max_dl']}")
    print(f"Correlation: {results[lang]['corr']:.3f}")
    print(f"Avg Height: {results[lang]['avg_height']:.3f}")


# =============================================================================
# COMPARISON PRINT
# =============================================================================

print("\n" + "="*70)
print("COMPARISON")
print("="*70)

for lang, r in results.items():
    print(f"\n{lang}")
    print(f"Avg DL: {r['avg_dl']:.3f}")
    print(f"Max DL: {r['max_dl']}")
    print(f"Std DL: {r['std_dl']:.3f}")
    print(f"90th %: {r['p90_dl']:.3f}")
    print(f"Correlation: {r['corr']:.3f}")
    print(f"Avg Height: {r['avg_height']:.3f}")


langs = list(results.keys())


# =============================================================================
# VISUALIZATIONS
# =============================================================================

# 1. Avg DL Bar
plt.figure()
plt.bar(langs, [results[l]["avg_dl"] for l in langs])
plt.title("Average Dependency Length")
plt.xticks(rotation=30)
plt.show()


# 2. Boxplot
plt.figure()
plt.boxplot([results[l]["distribution"] for l in langs], labels=langs)
plt.title("Dependency Length Distribution (Boxplot)")
plt.xticks(rotation=30)
plt.show()


# 3. Sentence Length vs DL
plt.figure()
for lang, path in languages.items():
    x, y = sentence_length_vs_dl(path)
    plt.scatter(x, y, alpha=0.3, label=lang)

plt.legend()
plt.title("Sentence Length vs Dependency Length")
plt.xlabel("Sentence Length")
plt.ylabel("Avg DL")
plt.show()


# 4. Tree Height Bar
plt.figure()
plt.bar(langs, [results[l]["avg_height"] for l in langs])
plt.title("Average Tree Height")
plt.xticks(rotation=30)
plt.show()


# 5. Correlation Bar
plt.figure()
plt.bar(langs, [results[l]["corr"] for l in langs])
plt.title("Correlation (Sentence Length vs DL)")
plt.xticks(rotation=30)
plt.show()


# 6. Cumulative Distribution
plt.figure()
for lang in langs:
    data = np.sort(results[lang]["distribution"])
    y = np.arange(len(data)) / len(data)
    plt.plot(data, y, label=lang)

plt.legend()
plt.title("Cumulative Distribution of DL")
plt.xlabel("Dependency Length")
plt.ylabel("CDF")
plt.show()


# 7. Heatmap
df = pd.DataFrame({
    "Language": langs,
    "Avg DL": [results[l]["avg_dl"] for l in langs],
    "Max DL": [results[l]["max_dl"] for l in langs],
    "Avg Height": [results[l]["avg_height"] for l in langs],
    "Correlation": [results[l]["corr"] for l in langs]
})

df.set_index("Language", inplace=True)

plt.figure(figsize=(8,5))
sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Cross-Language Comparison Heatmap")
plt.show()


print("\n" + "="*70)
print("DONE")
print("="*70)