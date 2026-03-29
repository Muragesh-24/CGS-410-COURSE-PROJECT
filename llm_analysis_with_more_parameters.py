import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from scipy.stats import pearsonr

print("Loading GPT-2...")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained(
    "gpt2",
    output_attentions=True
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)

model.eval()

print("Model loaded")

# ---------------------------
# Load dataset
# ---------------------------

df = pd.read_csv(
    "data/synthetic_sentences_recursive_depths.csv"
)

print("Total sentences:",len(df))


# ---------------------------
# Core metric function
# ---------------------------

def analyze_sentence(sentence):

    inputs = tokenizer(
        sentence,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():

        outputs = model(**inputs)

    logits = outputs.logits

    shift_logits = logits[:,:-1,:]

    shift_labels = inputs["input_ids"][:,1:]

    log_probs = torch.nn.functional.log_softmax(
        shift_logits,
        dim=-1
    )

    token_log_probs = log_probs.gather(
        2,
        shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    token_surprisal = -token_log_probs / np.log(2)

    avg_surprisal = token_surprisal.mean().item()

    max_surprisal = token_surprisal.max().item()

    std_surprisal = token_surprisal.std().item()

    perplexity = torch.exp(
        -token_log_probs.mean()
    ).item()

    # prediction entropy

    probs = torch.exp(log_probs)

    entropy = -(probs * log_probs).sum(
        dim=-1
    ).mean().item()

    return {

    "avg_surprisal":avg_surprisal,

    "max_surprisal":max_surprisal,

    "std_surprisal":std_surprisal,

    "perplexity":perplexity,

    "entropy":entropy,

    "token_surprisals":token_surprisal.cpu().numpy()[0],

    "tokens":tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

}


# ---------------------------
# Run analysis
# ---------------------------

results = []

for i,row in df.iterrows():

    metrics = analyze_sentence(
        row['sentence']
    )

    metrics['depth'] = row['depth']

    metrics['sentence'] = row['sentence']

    metrics['tokens'] = metrics['tokens']

    metrics['token_surprisals'] = metrics['token_surprisals']

    results.append(metrics)

results_df = pd.DataFrame(results)

print("Analysis complete")


# ---------------------------
# Merge results
# ---------------------------

df = df.merge(
    results_df,
    on=['sentence','depth']
)

# sentence length

df['length'] = df['sentence'].apply(
    lambda x: len(x.split())
)


# ---------------------------
# Depth aggregation
# ---------------------------

depth_stats = df.groupby(
    'depth'
).agg({

    'avg_surprisal':'mean',

    'max_surprisal':'mean',

    'perplexity':'mean',

    'entropy':'mean',

    'length':'mean'

}).reset_index()


print(depth_stats)


# ---------------------------
# Plot 1
# ---------------------------

plt.figure(figsize=(10,6))

plt.plot(

    depth_stats['depth'],

    depth_stats['avg_surprisal'],

    marker='o',

    label="Average Surprisal"

)

plt.plot(

    depth_stats['depth'],

    depth_stats['max_surprisal'],

    marker='s',

    label="Maximum Token Surprisal"

)

plt.xlabel("Embedding depth")

plt.ylabel("Difficulty")

plt.title("Token difficulty vs syntactic depth")

plt.legend()

plt.grid()

plt.show()


# ---------------------------
# Plot 2
# ---------------------------

plt.figure(figsize=(10,6))

plt.plot(

    depth_stats['depth'],

    depth_stats['perplexity'],

    marker='o'

)

plt.xlabel("Embedding depth")

plt.ylabel("Sentence perplexity")

plt.title("Perplexity scaling with depth")

plt.grid()

plt.show()


# ---------------------------
# Plot 3
# ---------------------------

plt.figure(figsize=(10,6))

plt.plot(

    depth_stats['depth'],

    depth_stats['entropy'],

    marker='o'

)

plt.xlabel("Embedding depth")

plt.ylabel("Prediction entropy")

plt.title("Model uncertainty vs depth")

plt.grid()

plt.show()


# ---------------------------
# Correlations
# ---------------------------

print("\nCorrelations")

c1,_ = pearsonr(
    df['depth'],
    df['avg_surprisal']
)

c2,_ = pearsonr(
    df['depth'],
    df['max_surprisal']
)

c3,_ = pearsonr(
    df['depth'],
    df['entropy']
)

c4,_ = pearsonr(
    df['depth'],
    df['perplexity']
)

print("Depth vs avg surprisal:",c1)

print("Depth vs max surprisal:",c2)

print("Depth vs entropy:",c3)

print("Depth vs perplexity:",c4)


# ---------------------------
# Hardest sentences
# ---------------------------

hard = df.sort_values(

    'max_surprisal',

    ascending=False

).head(15)


print("\nHardest sentences:\n")

for i,row in hard.iterrows():

    print(

        "Depth:",row['depth'],

        " Max surprisal:",

        round(row['max_surprisal'],2)

    )

    print(row['sentence'])

    print()


# ---------------------------
# Save results
# ---------------------------

df.to_csv(

    "llm_depth_analysis_results.csv",

    index=False

)

print("Saved results file")



def get_main_verb_surprisal(row):

    tokens = row['tokens']

    surprisals = row['token_surprisals']

    # crude estimate: main verb usually near end
    index = len(surprisals) - 2

    if index >= 0:
        return surprisals[index]

    return np.nan


df['main_verb_surprisal'] = df.apply(
    get_main_verb_surprisal,
    axis=1
)

main_stats = df.groupby(
    'depth'
)['main_verb_surprisal'].mean().reset_index()

plt.figure(figsize=(10,6))

plt.plot(

    main_stats['depth'],

    main_stats['main_verb_surprisal'],

    marker='o',

    color='red'

)

plt.xlabel("Embedding depth")

plt.ylabel("Main verb surprisal")

plt.title("Main Verb Prediction Difficulty vs Depth")

plt.grid()

plt.show()


def hardest_token(row):

    surprisals = row['token_surprisals']

    tokens = row['tokens']

    ignore = ['The','ĠThe','Ġthat','Ġis','.']
    valid = []
    for i,t in enumerate(tokens):
        if t not in ignore:
            valid.append((t,surprisals[i]))
    
    if len(valid) == 0:
        return tokens[np.argmax(surprisals)]
    index = max(valid, key=lambda x: x[1])[0]
    return index


df['hardest_token'] = df.apply(
    hardest_token,
    axis=1
)


hard = df.sort_values(

    'max_surprisal',

    ascending=False

).head(10)

print("\nHardest sentences:\n")

for i,row in hard.iterrows():

    print("Depth:",row['depth'])

    print("Sentence:",row['sentence'])

    print("Hardest token:",row['hardest_token'])

    print("Max surprisal:",round(row['max_surprisal'],2))

    print()
    
    
    
    print("\nMost difficult tokens:")

print(

df['hardest_token'].value_counts().head(10)

)