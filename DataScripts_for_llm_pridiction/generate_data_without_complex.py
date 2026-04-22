import random
import pandas as pd

random.seed(42)

# Core components (same style, slightly expanded for diversity)
subjects = [
    "The boy", "The girl", "The dog", "The cat", "The teacher", "The student",
    "The scientist", "The doctor", "The artist", "The player"
]

verbs = [
    "saw", "chased", "liked", "helped", "scolded", "admired",
    "followed", "met", "called", "watched"
]

objects = [
    "the dog", "the cat", "the boy", "the girl", "the teacher",
    "the student", "the scientist", "the doctor", "the artist"
]

actions = [
    "runs fast", "is barking", "is studying", "is sleeping",
    "jumps high", "is reading", "is laughing", "is walking"
]

def generate_sentence(depth):
    """
    Simple chaining structure (NO complex embedding)
    Example:
    Depth 2 → The dog that the cat chased that the boy saw runs fast.
    """
    main_subject = random.choice(subjects)
    main_action = random.choice(actions)

    if depth == 0:
        return f"{main_subject} {main_action}."

    clause = main_subject
    for _ in range(depth):
        obj = random.choice(objects)
        verb = random.choice(verbs)
        clause = f"{clause} that {obj} {verb}"

    return f"{clause} {main_action}."

# 10,000 sentence distribution (balanced but slightly tapered)
depth_distribution = {
    0: 1500,
    1: 1500,
    2: 1500,
    3: 1200,
    4: 1100,
    5: 900,
    6: 800,
    7: 700,
    8: 800   # total = 10,000
}

all_sentences = []
seen = set()

for depth, n_sentences in depth_distribution.items():
    count = 0
    attempts = 0

    while count < n_sentences and attempts < n_sentences * 20:
        sentence = generate_sentence(depth)
        attempts += 1

        # Avoid duplicates
        if sentence not in seen:
            seen.add(sentence)
            all_sentences.append({
                "sentence": sentence,
                "depth": depth
            })
            count += 1

# Convert to DataFrame
df = pd.DataFrame(all_sentences)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv("synthetic_sentences_10000_simple_depths.csv", index=False)

print("Generated dataset size:", len(df))
print(df["depth"].value_counts().sort_index())
print("Saved to 'synthetic_sentences_10000_simple_depths.csv'")