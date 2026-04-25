import os
import math
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm

# =========================
# CONFIG
# =========================
INPUT_FILE = "data/synthetic_surprisal_dataset_10000.csv"
OUTPUT_DIR = "outputs/random_baseline"
TEXT_COLUMN = "sentence"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD DATA: HEADER NONE
# =========================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

df = pd.DataFrame(sentences, columns=[TEXT_COLUMN])

print(f"Loaded {len(df)} sentences")

print(f"Loaded {len(df)} sentences")

# =========================
# LOAD GPT-2
# =========================
print("Loading GPT-2...")
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()

vocab_size = tokenizer.vocab_size
random_token_surprisal = math.log2(vocab_size)

print(f"Device: {device}")
print(f"GPT-2 vocab size: {vocab_size}")
print(f"Random baseline surprisal: {random_token_surprisal:.4f} bits")

# =========================
# FUNCTIONS
# =========================
def gpt2_average_surprisal(sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    if input_ids.shape[1] < 2:
        return None

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    token_surprisals = -token_log_probs / math.log(2)
    return token_surprisals.mean().item()


# =========================
# COMPUTE RESULTS
# =========================
results = []

print("Computing GPT-2 surprisal and random baseline...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    sentence = row[TEXT_COLUMN]

    gpt2_surp = gpt2_average_surprisal(sentence)

    if gpt2_surp is None:
        continue

    results.append({
        "sentence": sentence,
        "token_count": len(tokenizer.encode(sentence)),
        "gpt2_avg_surprisal": gpt2_surp,
        "random_baseline_surprisal": random_token_surprisal,
        "surprisal_gap_random_minus_gpt2": random_token_surprisal - gpt2_surp,
        "gpt2_beats_random": gpt2_surp < random_token_surprisal
    })

out_df = pd.DataFrame(results)

csv_path = os.path.join(OUTPUT_DIR, "random_baseline_comparison.csv")
out_df.to_csv(csv_path, index=False)

print(f"\nSaved CSV: {csv_path}")

# =========================
# SUMMARY
# =========================
avg_gpt2 = out_df["gpt2_avg_surprisal"].mean()
avg_random = random_token_surprisal
beat_percent = out_df["gpt2_beats_random"].mean() * 100

print("\n========== RANDOM BASELINE SUMMARY ==========")
print(f"Total sentences analyzed: {len(out_df)}")
print(f"Average GPT-2 surprisal: {avg_gpt2:.4f} bits")
print(f"Random baseline surprisal: {avg_random:.4f} bits")
print(f"Average gap: {avg_random - avg_gpt2:.4f} bits")
print(f"GPT-2 beats random in: {beat_percent:.2f}% sentences")

summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Random Baseline Surprisal Summary\n")
    f.write("=================================\n")
    f.write(f"Total sentences analyzed: {len(out_df)}\n")
    f.write(f"Average GPT-2 surprisal: {avg_gpt2:.4f} bits\n")
    f.write(f"Random baseline surprisal: {avg_random:.4f} bits\n")
    f.write(f"Average gap: {avg_random - avg_gpt2:.4f} bits\n")
    f.write(f"GPT-2 beats random in: {beat_percent:.2f}% sentences\n")

print(f"Saved summary: {summary_path}")

# =========================
# PLOT 1: BAR COMPARISON
# =========================
plt.figure(figsize=(7, 5))
plt.bar(
    ["GPT-2", "Random Baseline"],
    [avg_gpt2, avg_random]
)
plt.ylabel("Average Token Surprisal (bits)")
plt.title("GPT-2 vs Random Baseline")
plt.tight_layout()

plot1 = os.path.join(OUTPUT_DIR, "01_gpt2_vs_random_bar.png")
plt.savefig(plot1, dpi=300)
plt.close()

# =========================
# PLOT 2: HISTOGRAM
# =========================
plt.figure(figsize=(8, 5))
plt.hist(out_df["gpt2_avg_surprisal"], bins=40, alpha=0.8)
plt.axvline(avg_random, linestyle="--", linewidth=2, label="Random Baseline")
plt.xlabel("GPT-2 Average Token Surprisal (bits)")
plt.ylabel("Number of Sentences")
plt.title("Distribution of GPT-2 Surprisal")
plt.legend()
plt.tight_layout()

plot2 = os.path.join(OUTPUT_DIR, "02_gpt2_surprisal_distribution.png")
plt.savefig(plot2, dpi=300)
plt.close()

# =========================
# PLOT 3: TOKEN COUNT VS SURPRISAL
# =========================
plt.figure(figsize=(8, 5))
plt.scatter(
    out_df["token_count"],
    out_df["gpt2_avg_surprisal"],
    alpha=0.4,
    s=10
)
plt.axhline(avg_random, linestyle="--", linewidth=2, label="Random Baseline")
plt.xlabel("Token Count")
plt.ylabel("GPT-2 Average Token Surprisal (bits)")
plt.title("Sentence Length vs GPT-2 Surprisal")
plt.legend()
plt.tight_layout()

plot3 = os.path.join(OUTPUT_DIR, "03_token_count_vs_surprisal.png")
plt.savefig(plot3, dpi=300)
plt.close()

# =========================
# PLOT 4: GAP DISTRIBUTION
# =========================
plt.figure(figsize=(8, 5))
plt.hist(out_df["surprisal_gap_random_minus_gpt2"], bins=40, alpha=0.8)
plt.axvline(
    out_df["surprisal_gap_random_minus_gpt2"].mean(),
    linestyle="--",
    linewidth=2,
    label="Mean Gap"
)
plt.xlabel("Random Baseline Surprisal - GPT-2 Surprisal")
plt.ylabel("Number of Sentences")
plt.title("How Much GPT-2 Beats Random Baseline")
plt.legend()
plt.tight_layout()

plot4 = os.path.join(OUTPUT_DIR, "04_random_minus_gpt2_gap.png")
plt.savefig(plot4, dpi=300)
plt.close()

print("\nSaved plots:")
print(plot1)
print(plot2)
print(plot3)
print(plot4)
print("\nDone.")