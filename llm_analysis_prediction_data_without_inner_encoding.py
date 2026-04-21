import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "gpt2"
INPUT_CSV = "data/synthetic_sentences_10000_simple_depths.csv"
OUTPUT_CSV = "synthetic_sentences_with_prediction.csv"
BATCH_SIZE = 32   # reduce to 16 if memory issues
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading GPT-2 tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"Using device: {device}")

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

if "sentence" not in df.columns:
    raise ValueError("CSV must contain a 'sentence' column")

if "depth" not in df.columns:
    raise ValueError("CSV must contain a 'depth' column")

print(f"Loaded {len(df)} sentences")

# ============================================================
# BATCH SURPRISAL FUNCTION
# ============================================================

def compute_surprisal_batch(sentences):
    """
    Computes:
    - average token surprisal (bits)
    - max token surprisal (bits)
    - sentence perplexity

    Returns lists for each sentence in batch.
    """
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    # Log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather actual token log probs
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Convert to surprisal in bits
    token_surprisal = -token_log_probs / np.log(2)

    avg_surprisals = []
    max_surprisals = []
    perplexities = []

    for i in range(len(sentences)):
        valid_mask = shift_mask[i] == 1

        valid_log_probs = token_log_probs[i][valid_mask]
        valid_surprisal = token_surprisal[i][valid_mask]

        avg_surprisal = valid_surprisal.mean().item()
        max_surprisal = valid_surprisal.max().item()
        perplexity = torch.exp(-valid_log_probs.mean()).item()

        avg_surprisals.append(avg_surprisal)
        max_surprisals.append(max_surprisal)
        perplexities.append(perplexity)

    return avg_surprisals, max_surprisals, perplexities

# ============================================================
# RUN IN BATCHES
# ============================================================

all_avg_surprisals = []
all_max_surprisals = []
all_perplexities = []

sentences = df["sentence"].astype(str).tolist()
total = len(sentences)

print("Starting batched inference...")

for start_idx in range(0, total, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, total)
    batch_sentences = sentences[start_idx:end_idx]

    avg_s, max_s, ppl = compute_surprisal_batch(batch_sentences)

    all_avg_surprisals.extend(avg_s)
    all_max_surprisals.extend(max_s)
    all_perplexities.extend(ppl)

    print(f"Processed {end_idx}/{total}")

# Add results to dataframe
df["surprisal"] = all_avg_surprisals
df["max_surprisal"] = all_max_surprisals
df["perplexity"] = all_perplexities
df["length"] = df["sentence"].apply(lambda x: len(str(x).split()))

# Save full output
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved results to: {OUTPUT_CSV}")

# ============================================================
# AGGREGATION
# ============================================================

agg_mean = df.groupby("depth")[["surprisal", "max_surprisal", "perplexity"]].mean().reset_index()
agg_std = df.groupby("depth")[["surprisal", "max_surprisal", "perplexity"]].std().reset_index()

print("\nMean metrics by depth:")
print(agg_mean)

# ============================================================
# PLOT 1: AVG SURPRISAL VS DEPTH
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(agg_mean["depth"], agg_mean["surprisal"], marker="o")
plt.xlabel("Embedding Depth")
plt.ylabel("Average Surprisal (bits)")
plt.title("GPT-2 Average Surprisal vs Embedding Depth")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 2: PERPLEXITY VS DEPTH
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(agg_mean["depth"], agg_mean["perplexity"], marker="s")
plt.xlabel("Embedding Depth")
plt.ylabel("Perplexity")
plt.title("GPT-2 Perplexity vs Embedding Depth")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 3: MAX SURPRISAL VS DEPTH
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(agg_mean["depth"], agg_mean["max_surprisal"], marker="^")
plt.xlabel("Embedding Depth")
plt.ylabel("Max Token Surprisal (bits)")
plt.title("GPT-2 Max Token Surprisal vs Embedding Depth")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 4: ERROR BAR PLOT FOR SURPRISAL
# ============================================================

plt.figure(figsize=(10, 6))
plt.errorbar(
    agg_mean["depth"],
    agg_mean["surprisal"],
    yerr=agg_std["surprisal"],
    fmt="o-",
    capsize=5,
    label="Average Surprisal"
)
plt.xlabel("Embedding Depth")
plt.ylabel("Average Surprisal (bits)")
plt.title("GPT-2 Average Surprisal vs Embedding Depth (with Std Dev)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 5: ERROR BAR PLOT FOR PERPLEXITY
# ============================================================

plt.figure(figsize=(10, 6))
plt.errorbar(
    agg_mean["depth"],
    agg_mean["perplexity"],
    yerr=agg_std["perplexity"],
    fmt="s--",
    capsize=5,
    label="Perplexity"
)
plt.xlabel("Embedding Depth")
plt.ylabel("Perplexity")
plt.title("GPT-2 Perplexity vs Embedding Depth (with Std Dev)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 6: HISTOGRAM OF SURPRISAL
# ============================================================

plt.figure(figsize=(10, 6))
plt.hist(df["surprisal"], bins=30, edgecolor="black")
plt.xlabel("Average Surprisal (bits)")
plt.ylabel("Number of Sentences")
plt.title("Distribution of Average Surprisal")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 7: HISTOGRAM OF PERPLEXITY
# ============================================================

plt.figure(figsize=(10, 6))
plt.hist(df["perplexity"], bins=30, edgecolor="black")
plt.xlabel("Perplexity")
plt.ylabel("Number of Sentences")
plt.title("Distribution of Perplexity")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 8: HISTOGRAM OF MAX SURPRISAL
# ============================================================

plt.figure(figsize=(10, 6))
plt.hist(df["max_surprisal"], bins=30, edgecolor="black")
plt.xlabel("Max Token Surprisal (bits)")
plt.ylabel("Number of Sentences")
plt.title("Distribution of Max Token Surprisal")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 9: SURPRISAL VS SENTENCE LENGTH
# ============================================================

plt.figure(figsize=(10, 6))
plt.scatter(df["length"], df["surprisal"], alpha=0.5)
plt.xlabel("Sentence Length (words)")
plt.ylabel("Average Surprisal (bits)")
plt.title("Average Surprisal vs Sentence Length")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 10: PERPLEXITY VS SENTENCE LENGTH
# ============================================================

plt.figure(figsize=(10, 6))
plt.scatter(df["length"], df["perplexity"], alpha=0.5)
plt.xlabel("Sentence Length (words)")
plt.ylabel("Perplexity")
plt.title("Perplexity vs Sentence Length")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 11: BOXPLOT OF SURPRISAL BY DEPTH
# ============================================================

plt.figure(figsize=(10, 6))
df.boxplot(column="surprisal", by="depth", grid=True)
plt.xlabel("Embedding Depth")
plt.ylabel("Average Surprisal (bits)")
plt.title("Average Surprisal by Embedding Depth")
plt.suptitle("")
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 12: BOXPLOT OF PERPLEXITY BY DEPTH
# ============================================================

plt.figure(figsize=(10, 6))
df.boxplot(column="perplexity", by="depth", grid=True)
plt.xlabel("Embedding Depth")
plt.ylabel("Perplexity")
plt.title("Perplexity by Embedding Depth")
plt.suptitle("")
plt.tight_layout()
plt.show()

# ============================================================
# PLOT 13: BOXPLOT OF MAX SURPRISAL BY DEPTH
# ============================================================

plt.figure(figsize=(10, 6))
df.boxplot(column="max_surprisal", by="depth", grid=True)
plt.xlabel("Embedding Depth")
plt.ylabel("Max Token Surprisal (bits)")
plt.title("Max Token Surprisal by Embedding Depth")
plt.suptitle("")
plt.tight_layout()
plt.show()

# ============================================================
# HARDEST SENTENCES
# ============================================================

hardest = df.sort_values("surprisal", ascending=False).head(10)

print("\nTop 10 hardest sentences for GPT-2 (by average surprisal):\n")
for _, row in hardest.iterrows():
    print(
        f"Depth {row['depth']} | "
        f"Length {row['length']} | "
        f"Surprisal {row['surprisal']:.2f} bits | "
        f"Perplexity {row['perplexity']:.2f}\n"
        f"{row['sentence']}\n"
    )

# ============================================================
# OPTIONAL SUMMARY STATS
# ============================================================

print("\nOverall summary:")
print(df[["surprisal", "max_surprisal", "perplexity", "length"]].describe())

print("\nDone.")