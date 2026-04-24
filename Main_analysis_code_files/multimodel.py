import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# SETTINGS
# ============================================================

INPUT_CSV = "../data/synthetic_sentences_recursive_depths_10000_without_cues.csv"
TEXT_COL = "sentence"
DEPTH_COL = "depth"

BATCH_SIZE = 16
MAX_ROWS = 2000   

MODELS = {
    "GPT-2": "gpt2",
    "DistilGPT-2": "distilgpt2",
    "GPT-2 Medium": "gpt2-medium"
}

OUT_DIR = "multi_model_attention_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

if MAX_ROWS:
    df = df.sample(min(MAX_ROWS, len(df)), random_state=42).reset_index(drop=True)

sentences = df[TEXT_COL].astype(str).tolist()
depths = df[DEPTH_COL].tolist()


# ============================================================
# SURPRISAL FUNCTION
# ============================================================

def compute_surprisal_batch(model, tokenizer, batch_sentences):
    inputs = tokenizer(
        batch_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_surprisal = -token_log_probs / np.log(2)

    avg_s = []
    max_s = []

    for i in range(len(batch_sentences)):
        valid = shift_mask[i] == 1
        valid_surprisal = token_surprisal[i][valid]

        avg_s.append(valid_surprisal.mean().item())
        max_s.append(valid_surprisal.max().item())

    return avg_s, max_s


# ============================================================
# MULTI-MODEL ANALYSIS
# ============================================================

all_results = []

for model_label, model_name in MODELS.items():
    print(f"\nLoading model: {model_label}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    model_avg = []
    model_max = []

    for start in range(0, len(sentences), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(sentences))
        batch = sentences[start:end]

        avg_s, max_s = compute_surprisal_batch(model, tokenizer, batch)

        model_avg.extend(avg_s)
        model_max.extend(max_s)

        print(f"{model_label}: processed {end}/{len(sentences)}")

    temp = pd.DataFrame({
        "model": model_label,
        "depth": depths,
        "sentence": sentences,
        "avg_surprisal": model_avg,
        "max_surprisal": model_max
    })

    all_results.append(temp)

    del model
    torch.cuda.empty_cache()


result_df = pd.concat(all_results, ignore_index=True)
result_df.to_csv(os.path.join(OUT_DIR, "multi_model_surprisal_results.csv"), index=False)

agg = result_df.groupby(["model", "depth"])[["avg_surprisal", "max_surprisal"]].mean().reset_index()


# ============================================================
# PLOT 1: MULTI-MODEL AVG SURPRISAL
# ============================================================

plt.figure(figsize=(7, 5))

for model_label in MODELS.keys():
    sub = agg[agg["model"] == model_label]
    plt.plot(sub["depth"], sub["avg_surprisal"], marker="o", linewidth=2, label=model_label)

plt.xlabel("Embedding Depth")
plt.ylabel("Average Surprisal (bits)")
plt.title("Multi-Model Comparison: Average Surprisal vs Depth")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "multi_model_avg_surprisal.png"), dpi=300)
plt.show()


# ============================================================
# PLOT 2: MULTI-MODEL MAX SURPRISAL
# ============================================================

plt.figure(figsize=(7, 5))

for model_label in MODELS.keys():
    sub = agg[agg["model"] == model_label]
    plt.plot(sub["depth"], sub["max_surprisal"], marker="o", linewidth=2, label=model_label)

plt.xlabel("Embedding Depth")
plt.ylabel("Maximum Token Surprisal (bits)")
plt.title("Multi-Model Comparison: Max Surprisal vs Depth")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "multi_model_max_surprisal.png"), dpi=300)
plt.show()


# ============================================================
# ATTENTION ANALYSIS
# ============================================================

ATTENTION_MODEL = "gpt2"
print("\nLoading GPT-2 for attention analysis...")

tokenizer = AutoTokenizer.from_pretrained(ATTENTION_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    ATTENTION_MODEL,
    output_attentions=True,
    attn_implementation="eager"
)

model.to(device)
model.eval()


def get_attention_matrix(sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # last layer attention, averaged over heads
    attention = outputs.attentions[-1][0].mean(dim=0).detach().cpu().numpy()

    return tokens, attention


# Pick one medium-depth sentence for heatmap
sample_row = df.iloc[len(df) // 2]
sample_sentence = str(sample_row[TEXT_COL])

tokens, attention = get_attention_matrix(sample_sentence)

plt.figure(figsize=(8, 6))
plt.imshow(attention, aspect="auto")
plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=7)
plt.yticks(range(len(tokens)), tokens, fontsize=7)
plt.colorbar(label="Attention Weight")
plt.title("GPT-2 Attention Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "attention_heatmap.png"), dpi=300)
plt.show()


# ============================================================
# ATTENTION TREND: FINAL TOKEN ATTENTION TO EARLY TOKENS
# ============================================================

attention_rows = []

for depth in sorted(df[DEPTH_COL].unique()):
    sub = df[df[DEPTH_COL] == depth].head(20)

    scores = []

    for sentence in sub[TEXT_COL].astype(str):
        tokens, attn = get_attention_matrix(sentence)

        if len(tokens) < 4:
            continue

        # final token attending to first 25% of sentence
        final_token_index = len(tokens) - 1
        early_region_end = max(1, len(tokens) // 4)

        score = attn[final_token_index, :early_region_end].mean()
        scores.append(score)

    if scores:
        attention_rows.append({
            "depth": depth,
            "early_context_attention": np.mean(scores)
        })


attention_df = pd.DataFrame(attention_rows)
attention_df.to_csv(os.path.join(OUT_DIR, "attention_depth_summary.csv"), index=False)

plt.figure(figsize=(7, 5))
plt.plot(
    attention_df["depth"],
    attention_df["early_context_attention"],
    marker="o",
    linewidth=2
)

plt.xlabel("Embedding Depth")
plt.ylabel("Attention to Early Context")
plt.title("Attention to Earlier Tokens Across Depth")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "attention_early_context_vs_depth.png"), dpi=300)
plt.show()


print("\nSaved outputs:")
print("1. multi_model_avg_surprisal.png")
print("2. multi_model_max_surprisal.png")
print("3. attention_heatmap.png")
print("4. attention_early_context_vs_depth.png")
print("5. multi_model_surprisal_results.csv")
print("6. attention_depth_summary.csv")