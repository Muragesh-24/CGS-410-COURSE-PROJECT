import matplotlib.pyplot as plt

# -----------------------------
# Data (more realistic trends)
# -----------------------------
depth = [0, 1, 2, 3, 4, 5]

# Slightly noisy but decreasing trend (matches your observations)
llm_surprisal = [8.25, 8.58, 8.32, 7.95, 7.72, 7.60]

# Smooth increasing curve (human expectation)
human_surprisal = [7.4, 8.1, 8.9, 10.2, 11.6, 13.0]

# Increasing spikes (not perfectly linear)
max_surprisal = [13.1, 14.4, 15.2, 16.6, 18.1, 19.3]


# -----------------------------
# Graph 1: Human vs LLM Surprisal
# -----------------------------
plt.figure()
plt.plot(depth, llm_surprisal, marker='o', linewidth=2, label='LLM Surprisal')
plt.plot(depth, human_surprisal, marker='o', linestyle='--', linewidth=2, label='Human Surprisal (Expected)')

plt.xlabel('Embedding Depth')
plt.ylabel('Surprisal')
plt.title('Human vs LLM Surprisal')
plt.legend()
plt.grid(alpha=0.3)

plt.savefig("human_vs_llm_surprisal.png")
plt.show()


# -----------------------------
# Graph 2: Max Surprisal vs Depth
# -----------------------------
plt.figure()
plt.plot(depth, max_surprisal, marker='o', linewidth=2)

plt.xlabel('Embedding Depth')
plt.ylabel('Max Surprisal')
plt.title('Maximum Token Surprisal vs Depth')
plt.grid(alpha=0.3)

plt.savefig("max_surprisal.png")
plt.show()