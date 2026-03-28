import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load your CSV
df = pd.read_csv("data/synthetic_sentences_depths.csv")

def compute_surprisal(sentence):
    """
    Computes token-level surprisal and sentence-level perplexity.
    Returns: average token surprisal (bits) and perplexity.
    """
    # tokens = tokenizer.encode(sentence, return_tensors="pt").to(device)
    # with torch.no_grad():
    #     outputs = model(tokens, labels=tokens)
    #     loss = outputs.loss.item()  # average negative log likelihood over tokens
    #     perplexity = torch.exp(torch.tensor(loss)).item()
    #     # Token-level surprisal in bits
    #     surprisal_bits = loss * np.log2(np.e)  # convert nats to bits
    # return surprisal_bits, perplexity
    


    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = inputs["input_ids"][:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    token_log_probs = log_probs.gather(
        2,
        shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    token_surprisal = -token_log_probs / np.log(2)

    avg_surprisal = token_surprisal.mean().item()

    perplexity = torch.exp(-token_log_probs.mean()).item()

    return avg_surprisal, perplexity
    
    

# Apply to all sentences
surprisals = []
perplexities = []
for sentence in df['sentence']:
    s, p = compute_surprisal(sentence)
    surprisals.append(s)
    perplexities.append(p)

df['surprisal'] = surprisals
df['perplexity'] = perplexities

# Aggregate by depth
agg = df.groupby('depth')[['surprisal', 'perplexity']].mean().reset_index()

# Plot results
plt.figure(figsize=(10,6))
plt.plot(agg['depth'], agg['surprisal'], marker='o', label='Average Surprisal (bits)')
plt.plot(agg['depth'], agg['perplexity'], marker='s', label='Perplexity')
plt.xlabel("Embedding Depth")
plt.ylabel("Prediction Difficulty")
plt.title("GPT-2 Prediction Difficulty vs. Embedding Depth")
plt.grid(True)
plt.legend()
plt.show()

# Optional: save results
df.to_csv("synthetic_sentences_with_predictions.csv", index=False)
print("Prediction results saved to 'synthetic_sentences_with_predictions.csv'")

plt.figure(figsize=(10,6))
plt.hist(df['surprisal'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel("Surprisal (bits)")
plt.ylabel("Number of Sentences")
plt.title("Distribution of GPT-2 Surprisal Across Sentences")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
plt.hist(df['perplexity'], bins=30, color='salmon', edgecolor='black')
plt.xlabel("Perplexity")
plt.ylabel("Number of Sentences")
plt.title("Distribution of GPT-2 Perplexity Across Sentences")
plt.grid(True)
plt.show()


df['length'] = df['sentence'].apply(lambda x: len(x.split()))

plt.figure(figsize=(10,6))
plt.scatter(df['length'], df['surprisal'], alpha=0.6, color='purple')
plt.xlabel("Sentence Length (words)")
plt.ylabel("Average Surprisal (bits)")
plt.title("Surprisal vs. Sentence Length")
plt.grid(True)
plt.show()


agg = df.groupby('depth')[['surprisal', 'perplexity']].agg(['mean','std']).reset_index()

plt.figure(figsize=(10,6))
plt.errorbar(agg['depth'], agg['surprisal']['mean'], yerr=agg['surprisal']['std'], fmt='o-', label='Average Surprisal (bits)')
plt.errorbar(agg['depth'], agg['perplexity']['mean'], yerr=agg['perplexity']['std'], fmt='s--', label='Perplexity')
plt.xlabel("Embedding Depth")
plt.ylabel("Prediction Difficulty")
plt.title("GPT-2 Prediction Difficulty vs. Embedding Depth (with Std)")
plt.grid(True)
plt.legend()
plt.show()


hardest = df.sort_values('surprisal', ascending=False).head(10)
print("Top 10 hardest sentences for GPT-2:\n")
for i, row in hardest.iterrows():
    print(f"Depth {row['depth']}, Surprisal {row['surprisal']:.2f} bits: {row['sentence']}\n")