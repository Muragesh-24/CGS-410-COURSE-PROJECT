import random
import pandas as pd

random.seed(42)

# More lexical variety
subjects = [
    "The boy", "The girl", "The dog", "The cat", "The teacher", "The student",
    "The scientist", "The doctor", "The artist", "The lawyer", "The child",
    "The engineer", "The nurse", "The player", "The singer", "The manager",
    "The professor", "The actor", "The writer", "The farmer"
]

verbs = [
    "saw", "chased", "liked", "helped", "scolded", "admired", "called",
    "followed", "praised", "visited", "watched", "questioned", "ignored",
    "supported", "met", "blamed", "recognized", "encouraged"
]

objects = [
    "the dog", "the cat", "the boy", "the girl", "the teacher", "the student",
    "the scientist", "the doctor", "the artist", "the lawyer", "the child",
    "the engineer", "the nurse", "the player", "the singer", "the manager"
]

actions = [
    "runs fast", "is barking", "is studying", "is sleeping", "jumps high",
    "is reading", "is laughing", "is working", "is singing", "is smiling",
    "is writing", "is dancing", "is shouting", "is walking", "is resting"
]

adverbs = [
    "quickly", "silently", "happily", "sadly", "angrily", "carefully",
    "eagerly", "slowly", "proudly", "calmly"
]

# Optional modifiers for more variety
modifiers = [
    "",
    "from the village",
    "near the school",
    "in the garden",
    "with red shoes",
    "by the river",
    "from the city",
    "with great enthusiasm",
    "in the library",
    "during the morning"
]

def generate_inner(depth):
    """
    Recursively generates a subject phrase with nested relative clauses.
    """
    subj = random.choice(subjects)
    subj_mod = random.choice(modifiers)

    base = f"{subj} {subj_mod}".strip()

    if depth <= 0:
        return base
    else:
        verb = random.choice(verbs)
        inner_clause = generate_inner(depth - 1)
        return f"{base} that {inner_clause} {verb}"

def generate_sentence(depth):
    """
    Generate a sentence with recursive embedding and a main action.
    Keeps original casing — no lower()/capitalize().
    """
    clause = generate_inner(depth)
    main_action = random.choice(actions)
    adv = random.choice(adverbs)

    # several surface patterns for variety
    pattern = random.choice([
        f"{clause} {main_action}.",
        f"{clause} {main_action} {adv}.",
        f"{clause}, {main_action}.",
        f"{clause}, {main_action} {adv}."
    ])

    return " ".join(pattern.split())  # clean extra spaces only

# Target: 10,000 total
depth_distribution = {
    0: 2000,
    1: 2000,
    2: 2000,
    3: 1600,
    4: 1400,
    5: 1000
}

all_sentences = []
seen = set()

for depth, n_sentences in depth_distribution.items():
    count = 0
    attempts = 0

    while count < n_sentences and attempts < n_sentences * 20:
        sentence = generate_sentence(depth)
        attempts += 1

        # avoid duplicates
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

# Save CSV
df.to_csv("synthetic_sentences_recursive_depths_10000.csv", index=False)

print("Generated dataset size:", len(df))
print(df["depth"].value_counts().sort_index())
print("Saved to 'synthetic_sentences_recursive_depths_10000.csv'")