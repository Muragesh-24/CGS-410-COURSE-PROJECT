# import random
# import pandas as pd

# # Core components
# subjects = ["The boy", "The girl", "The dog", "The cat", "The teacher", "The student"]
# verbs = ["saw", "chased", "liked", "helped", "scolded", "admired"]
# objects = ["the dog", "the cat", "the boy", "the girl", "the teacher", "the student"]
# actions = ["runs fast", "is barking", "is studying", "is sleeping", "jumps high", "is reading"]

# def generate_inner(depth):
#     """
#     Recursively generates a subject phrase with nested relative clauses.
#     """
#     subj = random.choice(subjects)
#     if depth <= 0:
#         return subj
#     else:
#         obj = random.choice(objects)
#         verb = random.choice(verbs)
#         # Inner embedding
#         inner_clause = generate_inner(depth - 1)
#         return f"{subj} that {inner_clause} {verb}"

# def generate_sentence(depth):
#     """
#     Generates a full sentence with main action and recursive inner embeddings.
#     """
#     main_action = random.choice(actions)
#     clause = generate_inner(depth)
#     return f"{clause} {main_action}."

# # Decide number of sentences per depth
# depths = [0,1,2,3,4,5,6,7,8]
# sentences_per_depth = [40, 40, 40, 40, 30, 25, 25, 20, 20]  # total ~280

# all_sentences = []
# for depth, n_sentences in zip(depths, sentences_per_depth):
#     for _ in range(n_sentences):
#         sentence = generate_sentence(depth)
#         all_sentences.append({"sentence": sentence, "depth": depth})

# # Convert to DataFrame
# df = pd.DataFrame(all_sentences)

# # Shuffle
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Save CSV
# df.to_csv("synthetic_sentences_inner_depths.csv", index=False)
# print("Generated ~300 deeply nested sentences and saved to 'synthetic_sentences_inner_depths.csv'")

import random
import pandas as pd

# Core components for sentence generation
subjects = ["The boy", "The girl", "The dog", "The cat", "The teacher", "The student"]
verbs = ["saw", "chased", "liked", "helped", "scolded", "admired"]
objects = ["the dog", "the cat", "the boy", "the girl", "the teacher", "the student"]
actions = ["runs fast", "is barking", "is studying", "is sleeping", "jumps high", "is reading"]

def generate_inner(depth):
    """
    Recursively generates a subject phrase with nested relative clauses.
    Each inner clause can itself have a nested clause.
    """
    subj = random.choice(subjects)
    if depth <= 0:
        return subj
    else:
        # Pick an object for the relative clause
        obj = random.choice(objects)
        verb = random.choice(verbs)
        # Inner clause recursion
        inner_clause = generate_inner(depth - 1)
        return f"{subj} that {inner_clause} {verb}"

def generate_sentence(depth):
    """
    Generate a sentence with recursive inner embeddings and a main action.
    """
    main_action = random.choice(actions)
    clause = generate_inner(depth)
    # lower-case all content then capitalize only the first character
    sentence = f"{clause} {main_action}."
    return sentence.lower().capitalize()

# Decide number of sentences per depth
depths = [0,1,2,3,4,5]
sentences_per_depth = [40, 40, 40, 30, 25, 25]  # total ~200

all_sentences = []
for depth, n_sentences in zip(depths, sentences_per_depth):
    for _ in range(n_sentences):
        sentence = generate_sentence(depth)
        all_sentences.append({"sentence": sentence, "depth": depth})

# Convert to DataFrame
df = pd.DataFrame(all_sentences)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save CSV
df.to_csv("synthetic_sentences_recursive_depths.csv", index=False)
print("Generated ~200 deeply nested sentences and saved to 'synthetic_sentences_recursive_depths.csv'")




