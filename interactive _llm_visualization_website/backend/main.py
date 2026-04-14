from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math

app = Flask(__name__)
CORS(app)

print("Loading GPT-2...")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained(
    "gpt2",
    attn_implementation="eager"   
)


model.eval()

print("Model ready!")


def analyze(text):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
       outputs = model(
    **inputs,
    output_attentions=True   
)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

    attentions = outputs.attentions

    if attentions is not None and len(attentions) > 0:
          last_layer = attentions[-1][0]  # [heads, seq, seq]
          avg_attention = last_layer.mean(dim=0).tolist()
    else:
    # fallback (no attention)
       seq_len = len(tokens)
       avg_attention = [[0]*seq_len for _ in range(seq_len)]

    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    clean_tokens = [t.replace("Ġ", "") for t in tokens]

    results = []

    for i in range(len(tokens) - 1):
        next_token_id = input_ids[i + 1].item()
        prob = probs[0, i, next_token_id].item()
        surprisal = -math.log(prob + 1e-9)

        # Top-K predictions
        topk = torch.topk(probs[0, i], 5)
        candidates = []
        for idx, p in zip(topk.indices, topk.values):
            candidates.append({
                "token": tokenizer.decode([idx]).strip(),
                "prob": round(float(p), 4)
            })

        results.append({
            "token": clean_tokens[i + 1],
            "prob": round(prob, 4),
            "surprisal": round(surprisal, 4),
            "topk": candidates
        })

    # Average last layer attention across heads
    last_layer = attentions[-1][0]  # [heads, seq, seq]
    avg_attention = last_layer.mean(dim=0).tolist()

    return {
        "tokens": clean_tokens,
        "analysis": results,
        "attention": avg_attention
    }


@app.route("/analyze", methods=["POST"])
def api():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No input"}), 400

    return jsonify(analyze(text))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)