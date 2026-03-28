import random
import pandas as pd

# Core components to build sentences
subjects = ["The boy", "The girl", "The dog", "The cat", "The teacher", "The student"]
verbs = ["saw", "chased", "liked", "helped", "scolded", "admired"]
objects = ["the dog", "the cat", "the boy", "the girl", "the teacher", "the student"]
actions = ["runs fast", "is barking", "is studying", "is sleeping", "jumps high", "is reading"]

def generate_sentence(depth):
    """
    Generates a single English sentence with a given embedding depth.
    Depth 0 → simple sentence: "The dog runs fast."
    Depth 1 → single relative clause: "The dog that the cat chased runs fast."
    Depth n → nested relative clauses
    """
    # Start with the main subject
    main_subject = random.choice(subjects)
    main_action = random.choice(actions)

    if depth == 0:
        return f"{main_subject} {main_action}."
    
    # Build nested relative clauses
    clause = f"{main_subject}"
    for d in range(depth):
        obj = random.choice(objects)
        verb = random.choice(verbs)
        clause = f"{clause} that {obj} {verb}"
    
    return f"{clause} {main_action}."

# Decide number of sentences per depth
depths = [0,1,2,3,4,5,6,7,8]
sentences_per_depth = [40, 40, 40, 40, 30, 25, 25, 20, 20]  # total ~280

all_sentences = []
for depth, n_sentences in zip(depths, sentences_per_depth):
    for _ in range(n_sentences):
        sentence = generate_sentence(depth)
        all_sentences.append({"sentence": sentence, "depth": depth})

# Convert to DataFrame
df = pd.DataFrame(all_sentences)

# Shuffle to randomize order
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv("synthetic_sentences_depths.csv", index=False)
print("Generated ~300 sentences across multiple depths and saved to 'synthetic_sentences_depths.csv'")