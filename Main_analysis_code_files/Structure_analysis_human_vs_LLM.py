import os
import time
import logging
import pandas as pd
import spacy
from conllu import parse_incr
import matplotlib.pyplot as plt
# -----------------------------
# LOGGING SETUP
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOG_EVERY = 1000

# -----------------------------
# CONFIG
# -----------------------------

HUMAN_FILES = {
    "English": "data/Human_dependency_length/en_ewt-ud-dev.conllu.txt",
    "French": "data/Human_dependency_length/fr_gsd-ud-dev.conllu",
    "Spanish": "data/Human_dependency_length/es_ancora-ud-test.conllu",
    "Hindi": "data/Human_dependency_length/hi_hdtb-ud-dev.conllu",
    "Marathi": "data/Human_dependency_length/mr_ufal-ud-train.conllu",
    "German": "data/Human_dependency_length/de_hdt-ud-dev.conllu",
}

LLM_FILES = {
    "English_AIGTxt": "data/LLM_Dependency_Length/AIGTxt_dataset.csv",
    "French_LLM": "data/LLM_Dependency_Length/french.csv",
    "Hindi_LLM": "data/LLM_Dependency_Length/hindi.csv",
    "Spanish_LLM": "data/LLM_Dependency_Length/spanish.csv",
    "Kannada_LLM": "data/LLM_Dependency_Length/kannadda.csv",
    "English_txt": "data/llm_generated.txt",
}

OUTPUT_FILE = "tree_structure_dl_crossing_results.csv"
SUMMARY_FILE = "tree_structure_summary.csv"
PARTIAL_FILE = "tree_structure_partial_results.csv"

SPACY_MODELS = {
    "English": "en_core_web_sm",
    "French": "fr_core_news_sm",
    "Spanish": "es_core_news_sm",
    "German": "de_core_news_sm",
    "Hindi": "xx_ent_wiki_sm",
    "Marathi": "xx_ent_wiki_sm",
    "Kannada": "xx_ent_wiki_sm",
}

# -----------------------------
# BASIC METRICS
# -----------------------------

def dependency_lengths(heads):
    return [
        abs(i - h)
        for i, h in enumerate(heads, start=1)
        if h is not None and h != 0 and h != i
    ]


def count_crossings(heads):
    arcs = []

    for dep, head in enumerate(heads, start=1):
        if head is None or head == 0 or head == dep:
            continue

        a, b = sorted((dep, head))
        arcs.append((a, b))

    crossings = 0

    for i in range(len(arcs)):
        a, b = arcs[i]
        for j in range(i + 1, len(arcs)):
            c, d = arcs[j]

            if (a < c < b < d) or (c < a < d < b):
                crossings += 1

    return crossings


def tree_height_from_heads(heads):
    children = {i: [] for i in range(1, len(heads) + 1)}
    roots = []

    for dep, head in enumerate(heads, start=1):
        if head == 0 or head == dep or head is None:
            roots.append(dep)
        else:
            children.setdefault(head, []).append(dep)

    def height(node):
        if not children.get(node):
            return 1
        return 1 + max(height(child) for child in children[node])

    if not roots:
        return 0

    return max(height(root) for root in roots)


def compute_metrics(sentence, heads, source, language, dataset_type):
    lengths = dependency_lengths(heads)
    crossings = count_crossings(heads)

    return {
        "dataset_type": dataset_type,
        "source": source,
        "language": language,
        "sentence": sentence,
        "num_tokens": len(heads),
        "tree_height": tree_height_from_heads(heads),
        "avg_dependency_length": sum(lengths) / len(lengths) if lengths else 0,
        "max_dependency_length": max(lengths) if lengths else 0,
        "num_crossings": crossings,
        "has_crossing": crossings > 0,
    }

# -----------------------------
# HUMAN CONLLU ANALYSIS
# -----------------------------

def analyze_conllu_file(path, language):
    start_time = time.time()
    results = []
    processed = 0

    logging.info(f"[HUMAN START] {language} | file={path}")

    with open(path, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            words = []
            heads = []

            for token in tokenlist:
                token_id = token["id"]

                if not isinstance(token_id, int):
                    continue

                words.append(token["form"])
                heads.append(token["head"])

            if len(words) > 1:
                sentence = " ".join(words)

                results.append(
                    compute_metrics(
                        sentence=sentence,
                        heads=heads,
                        source=os.path.basename(path),
                        language=language,
                        dataset_type="Human corpus",
                    )
                )

            processed += 1

            if processed % LOG_EVERY == 0:
                elapsed = time.time() - start_time
                logging.info(
                    f"[HUMAN RUNNING] {language} | processed={processed} | "
                    f"valid={len(results)} | time={elapsed:.2f}s"
                )

    elapsed = time.time() - start_time
    logging.info(
        f"[HUMAN DONE] {language} | valid_sentences={len(results)} | "
        f"time={elapsed:.2f}s"
    )

    return results

# -----------------------------
# LLM RAW TEXT ANALYSIS
# -----------------------------

def get_spacy_model(language_name):
    base_lang = language_name.split("_")[0]
    model_name = SPACY_MODELS.get(base_lang, "xx_ent_wiki_sm")

    logging.info(f"[MODEL LOAD] {language_name} uses spaCy model: {model_name}")

    try:
        nlp = spacy.load(model_name)
        logging.info(f"[MODEL LOADED] {model_name}")
        return nlp
    except OSError:
        logging.error(f"[MODEL MISSING] {model_name}")
        logging.error(f"Install using: python -m spacy download {model_name}")
        return None


def extract_sentences_from_csv(path):
    logging.info(f"[CSV LOAD START] {path}")

    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "ISO-8859-1"]

    df = None
    used_encoding = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
            used_encoding = enc
            logging.info(f"[CSV LOAD SUCCESS] {path} | encoding={enc}")
            break
        except Exception as e:
            logging.warning(f"[CSV LOAD FAILED] {path} | encoding={enc} | error={e}")

    if df is None:
        raise ValueError(f"Could not read CSV file: {path}")

    possible_cols = [
        "sentence",
        "text",
        "Sentence",
        "Text",
        "generated_text",
        "output",
        "content",
    ]

    text_col = None

    for col in possible_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        text_col = df.columns[0]

    logging.info(
        f"[CSV COLUMN] {path} | selected_column={text_col} | "
        f"rows={len(df)} | encoding={used_encoding}"
    )

    sentences = (
        df[text_col]
        .dropna()
        .astype(str)
        .str.strip()
    )

    sentences = sentences[sentences != ""]

    logging.info(f"[CSV SENTENCES] {path} | usable_sentences={len(sentences)}")

    return sentences.tolist()


def extract_sentences_from_txt(path):
    logging.info(f"[TXT LOAD START] {path}")

    encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "ISO-8859-1"]

    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc) as f:
                sentences = [line.strip() for line in f if line.strip()]

            logging.info(
                f"[TXT LOAD SUCCESS] {path} | encoding={enc} | "
                f"sentences={len(sentences)}"
            )

            return sentences

        except UnicodeDecodeError:
            logging.warning(f"[TXT LOAD FAILED] {path} | encoding={enc}")

    raise ValueError(f"Could not read TXT file: {path}")


def analyze_raw_text_file(path, language):
    start_time = time.time()

    logging.info(f"[LLM START] {language} | file={path}")

    nlp = get_spacy_model(language)

    if nlp is None:
        logging.warning(f"[LLM SKIPPED] {language} because model was missing")
        return []

    if path.endswith(".csv"):
        sentences = extract_sentences_from_csv(path)
    else:
        sentences = extract_sentences_from_txt(path)

    logging.info(f"[LLM PARSE START] {language} | total_sentences={len(sentences)}")

    results = []

    for idx, sent in enumerate(sentences, start=1):
        try:
            doc = nlp(sent)

            words = []
            heads = []

            for token in doc:
                words.append(token.text)

                if token.head == token:
                    heads.append(0)
                else:
                    heads.append(token.head.i + 1)

            if len(words) > 1:
                results.append(
                    compute_metrics(
                        sentence=sent,
                        heads=heads,
                        source=os.path.basename(path),
                        language=language,
                        dataset_type="LLM corpus",
                    )
                )

        except Exception as e:
            logging.warning(
                f"[LLM SENTENCE SKIPPED] {language} | sentence_index={idx} | error={e}"
            )

        if idx % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            logging.info(
                f"[LLM RUNNING] {language} | processed={idx}/{len(sentences)} | "
                f"valid={len(results)} | time={elapsed:.2f}s"
            )

    elapsed = time.time() - start_time

    logging.info(
        f"[LLM DONE] {language} | valid_sentences={len(results)} | "
        f"time={elapsed:.2f}s"
    )

    return results

# -----------------------------
# SAVE HELPERS
# -----------------------------

def save_partial_results(all_results):
    if not all_results:
        return

    partial_df = pd.DataFrame(all_results)
    partial_df.to_csv(PARTIAL_FILE, index=False)

    logging.info(
        f"[PARTIAL SAVE] {PARTIAL_FILE} | rows={len(partial_df)}"
    )

def generate_comparison_graphs(df):
    os.makedirs("structure_graphs", exist_ok=True)

    # Simplify labels: Human corpus vs LLM corpus
    plot_df = df[df["dataset_type"].isin(["Human corpus", "LLM corpus"])].copy()

    metrics = {
        "avg_dependency_length": "Average Dependency Length",
        "tree_height": "Tree Height",
        "max_dependency_length": "Maximum Dependency Length",
        "num_crossings": "Number of Crossing Dependencies",
        "num_tokens": "Sentence Length / Tokens",
    }

    # 1. Boxplots: Human vs LLM for each metric
    for metric, title in metrics.items():
        plt.figure(figsize=(7, 5))
        plot_df.boxplot(column=metric, by="dataset_type")
        plt.title(f"Human vs LLM: {title}")
        plt.suptitle("")
        plt.xlabel("Dataset Type")
        plt.ylabel(title)
        plt.tight_layout()
        plt.savefig(f"structure_graphs/human_vs_llm_{metric}_boxplot.png", dpi=300)
        plt.close()

    # 2. Bar chart: average metrics by dataset type
    avg_df = (
        plot_df.groupby("dataset_type")
        .agg(
            avg_dependency_length=("avg_dependency_length", "mean"),
            avg_tree_height=("tree_height", "mean"),
            avg_max_dependency_length=("max_dependency_length", "mean"),
            avg_crossings=("num_crossings", "mean"),
            crossing_percent=("has_crossing", lambda x: 100 * x.mean()),
        )
        .reset_index()
    )

    for metric in [
        "avg_dependency_length",
        "avg_tree_height",
        "avg_max_dependency_length",
        "avg_crossings",
        "crossing_percent",
    ]:
        plt.figure(figsize=(7, 5))
        plt.bar(avg_df["dataset_type"], avg_df[metric])
        plt.title(f"Human vs LLM: {metric.replace('_', ' ').title()}")
        plt.xlabel("Dataset Type")
        plt.ylabel(metric.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(f"structure_graphs/human_vs_llm_{metric}_bar.png", dpi=300)
        plt.close()

    # 3. Scatter: dependency length vs tree height
    plt.figure(figsize=(7, 5))
    for dataset_type in plot_df["dataset_type"].unique():
        subset = plot_df[plot_df["dataset_type"] == dataset_type]
        plt.scatter(
            subset["tree_height"],
            subset["avg_dependency_length"],
            alpha=0.3,
            label=dataset_type,
            s=10,
        )

    plt.title("Tree Height vs Average Dependency Length")
    plt.xlabel("Tree Height")
    plt.ylabel("Average Dependency Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig("structure_graphs/tree_height_vs_avg_dependency_length.png", dpi=300)
    plt.close()

    # 4. Scatter: sentence length vs dependency length
    plt.figure(figsize=(7, 5))
    for dataset_type in plot_df["dataset_type"].unique():
        subset = plot_df[plot_df["dataset_type"] == dataset_type]
        plt.scatter(
            subset["num_tokens"],
            subset["avg_dependency_length"],
            alpha=0.3,
            label=dataset_type,
            s=10,
        )

    plt.title("Sentence Length vs Average Dependency Length")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Average Dependency Length")
    plt.legend()
    plt.tight_layout()
    plt.savefig("structure_graphs/sentence_length_vs_avg_dependency_length.png", dpi=300)
    plt.close()

    # 5. Language-wise comparison
    lang_summary = (
        plot_df.groupby(["dataset_type", "language"])
        .agg(
            avg_dependency_length=("avg_dependency_length", "mean"),
            avg_tree_height=("tree_height", "mean"),
            avg_crossings=("num_crossings", "mean"),
        )
        .reset_index()
    )

    for metric in ["avg_dependency_length", "avg_tree_height", "avg_crossings"]:
        pivot = lang_summary.pivot(
            index="language",
            columns="dataset_type",
            values=metric
        )

        plt.figure(figsize=(10, 5))
        pivot.plot(kind="bar", ax=plt.gca())
        plt.title(f"Language-wise Human vs LLM: {metric.replace('_', ' ').title()}")
        plt.xlabel("Language")
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"structure_graphs/language_wise_{metric}.png", dpi=300)
        plt.close()

    logging.info("[GRAPH SAVE] All graphs saved inside structure_graphs/")
def save_final_outputs(all_results):
    df = pd.DataFrame(all_results)

    if df.empty:
        logging.error("[FINAL ERROR] No results generated.")
        return

    generate_comparison_graphs(df)
   
    
    logging.info(f"[FINAL SAVE] {OUTPUT_FILE} | rows={len(df)}")

    summary = (
        df.groupby(["dataset_type", "language", "source"])
        .agg(
            sentences=("sentence", "count"),
            avg_tokens=("num_tokens", "mean"),
            avg_tree_height=("tree_height", "mean"),
            avg_dependency_length=("avg_dependency_length", "mean"),
            avg_max_dependency_length=("max_dependency_length", "mean"),
            avg_crossings=("num_crossings", "mean"),
            crossing_sentence_percent=("has_crossing", lambda x: 100 * x.mean()),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_FILE, index=False)
    logging.info(f"[SUMMARY SAVE] {SUMMARY_FILE} | rows={len(summary)}")

    logging.info("\n" + str(summary))

# -----------------------------
# MAIN
# -----------------------------

def main():
    total_start = time.time()
    all_results = []

    logging.info("=" * 70)
    logging.info("STRUCTURE ANALYSIS STARTED")
    logging.info("=" * 70)

    logging.info("Processing human datasets...")

    for language, path in HUMAN_FILES.items():
        if os.path.exists(path):
            results = analyze_conllu_file(path, language)
            all_results.extend(results)
            save_partial_results(all_results)
        else:
            logging.warning(f"[MISSING HUMAN FILE] {path}")

    logging.info("Processing LLM datasets...")

    for language, path in LLM_FILES.items():
        if os.path.exists(path):
            results = analyze_raw_text_file(path, language)
            all_results.extend(results)
            save_partial_results(all_results)
        else:
            logging.warning(f"[MISSING LLM FILE] {path}")

    save_final_outputs(all_results)

    total_elapsed = time.time() - total_start

    logging.info("=" * 70)
    logging.info(f"STRUCTURE ANALYSIS COMPLETED | total_rows={len(all_results)} | time={total_elapsed:.2f}s")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()