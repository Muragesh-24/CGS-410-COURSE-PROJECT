from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math
import time

app = Flask(__name__)
CORS(app)

print("Loading GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
model.eval()
print("Model ready!")

def analyze(text):
    start_time = time.time()

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    clean_tokens = [t.replace("Ġ", "") for t in tokens]

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    attentions = outputs.attentions

    results = []
    total_log_prob = 0.0

    for i in range(len(tokens) - 1):
        next_token_id = input_ids[i + 1].item()
        prob = probs[0, i, next_token_id].item()
        
        surprisal = -math.log(prob + 1e-9)
        total_log_prob += math.log(prob + 1e-9)

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

    sequence_length = len(tokens) - 1
    perplexity = math.exp(-total_log_prob / sequence_length) if sequence_length > 0 else 0

    if attentions is not None and len(attentions) > 0:
        last_layer = attentions[-1][0]
        avg_attention = last_layer.mean(dim=0).tolist()
    else:
        seq_len = len(tokens)
        avg_attention = [[0]*seq_len for _ in range(seq_len)]

    end_time = time.time()
    processing_time_ms = round((end_time - start_time) * 1000, 2)
    time_per_token_ms = round(processing_time_ms / len(tokens), 2) if len(tokens) > 0 else 0

    return {
        "tokens": clean_tokens,
        "analysis": results,
        "attention": avg_attention,
        "metrics": {
            "total_processing_time_ms": processing_time_ms,
            "token_count": len(tokens),
            "time_per_token_ms": time_per_token_ms,
            "overall_perplexity": round(perplexity, 4)
        }
    }

@app.route("/analyze", methods=["POST"])
def api():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No input text provided"}), 400
    return jsonify(analyze(text))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)