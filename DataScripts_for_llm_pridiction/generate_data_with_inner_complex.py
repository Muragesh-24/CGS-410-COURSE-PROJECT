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
# subjects = ["The boy", "The girl", "The dog", "The cat", "The teacher", "The student"]
# verbs = ["saw", "chased", "liked", "helped", "scolded", "admired"]
# objects = ["the dog", "the cat", "the boy", "the girl", "the teacher", "the student"]
# actions = ["runs fast", "is barking", "is studying", "is sleeping", "jumps high", "is reading"]

# def generate_inner(depth):
#     """
#     Recursively generates a subject phrase with nested relative clauses.
#     Each inner clause can itself have a nested clause.
#     """
#     subj = random.choice(subjects)
#     if depth <= 0:
#         return subj
#     else:
#         # Pick an object for the relative clause
#         obj = random.choice(objects)
#         verb = random.choice(verbs)
#         # Inner clause recursion
#         inner_clause = generate_inner(depth - 1)
#         return f"{subj} that {inner_clause} {verb}"

# def generate_sentence(depth):
#     """
#     Generate a sentence with recursive inner embeddings and a main action.
#     """
#     main_action = random.choice(actions)
#     clause = generate_inner(depth)
#     # lower-case all content then capitalize only the first character
#     sentence = f"{clause} {main_action}."
#     return sentence.lower().capitalize()

# # Decide number of sentences per depth
# depths = [0,1,2,3,4,5]
# sentences_per_depth = [40, 40, 40, 30, 25, 25]  # total ~200

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
# df.to_csv("synthetic_sentences_recursive_depths.csv", index=False)
# print("Generated ~200 deeply nested sentences and saved to 'synthetic_sentences_recursive_depths.csv'")





import random
import pandas as pd

random.seed(42)

# ============================================================
# LEXICON
# ============================================================

# Singular noun phrases
nouns = [
    "the boy", "the girl", "the dog", "the cat", "the teacher",
    "the student", "the doctor", "the lawyer", "the artist",
    "the singer", "the actor", "the engineer", "the farmer",
    "the child", "the mother", "the father", "the neighbor",
    "the driver", "the manager", "the scientist"
]

# Transitive verbs for relative clauses / matrix clauses
transitive_verbs = [
    "saw", "chased", "helped", "admired", "visited", "called",
    "followed", "met", "supported", "watched", "thanked",
    "praised", "blamed", "recognized", "questioned", "informed",
    "encouraged", "warned", "reminded", "greeted"
]

# Intransitive / predicate endings for the main clause
main_predicates = [
    "ran away",
    "laughed loudly",
    "slept peacefully",
    "worked quietly",
    "smiled warmly",
    "arrived early",
    "sat calmly",
    "waited outside",
    "spoke softly",
    "read carefully",
    "walked slowly",
    "rested briefly",
    "cried suddenly",
    "listened closely",
    "stood silently"
]

# Optional adverbs/adverbials to increase variety
adverbials = [
    "",
    "yesterday",
    "this morning",
    "at night",
    "during class",
    "after dinner",
    "before noon",
    "near the station",
    "in the garden",
    "at the office"
]

# Optional adjectives to increase lexical diversity
adjectives = [
    "",
    "young",
    "old",
    "tall",
    "quiet",
    "angry",
    "curious",
    "kind",
    "nervous",
    "happy",
    "serious",
    "clever"
]

# ============================================================
# HELPERS
# ============================================================

def maybe_add_adjective(np_text: str) -> str:
    """
    Converts 'the boy' -> 'the young boy' sometimes.
    """
    adj = random.choice(adjectives)
    if adj == "":
        return np_text

    parts = np_text.split()
    if parts[0] == "the":
        return "the " + adj + " " + " ".join(parts[1:])
    return np_text

def pick_distinct(exclude=None):
    """
    Pick a noun phrase different from `exclude` when possible.
    """
    candidates = nouns[:]
    if exclude in candidates and len(candidates) > 1:
        candidates.remove(exclude)
    return maybe_add_adjective(random.choice(candidates))

def build_relative_chain(head_np: str, depth: int) -> str:
    """
    Build center-embedded noun phrase like:
    'the boy that the girl saw'
    'the boy that the girl that the dog chased saw'
    etc.

    Structure:
    NP0 that NP1 that NP2 ... V2 V1
    """
    if depth == 0:
        return head_np

    embedded_subjects = []
    embedded_verbs = []

    previous = head_np
    for _ in range(depth):
        subj = pick_distinct(exclude=previous)
        verb = random.choice(transitive_verbs)
        embedded_subjects.append(subj)
        embedded_verbs.append(verb)
        previous = subj

    # Start with head noun phrase
    phrase = head_np

    # Add nested "that NP ..."
    for subj in embedded_subjects:
        phrase += f" that {subj}"

    # Close embeddings with reverse-order verbs
    for verb in reversed(embedded_verbs):
        phrase += f" {verb}"

    return phrase

def generate_sentence(depth: int) -> str:
    """
    Generate one sentence with a specific embedding depth.
    Example:
    Depth 0: The boy laughed loudly.
    Depth 1: The boy that the girl saw laughed loudly.
    Depth 2: The boy that the girl that the dog chased saw laughed loudly.
    """
    head = maybe_add_adjective(random.choice(nouns))
    subject_phrase = build_relative_chain(head, depth)
    predicate = random.choice(main_predicates)
    adv = random.choice(adverbials)

    if adv:
        sentence = f"{subject_phrase} {predicate} {adv}."
    else:
        sentence = f"{subject_phrase} {predicate}."

    return sentence[0].upper() + sentence[1:]

# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(
    total_sentences=10000,
    depth_distribution=None,
    output_csv="synthetic_surprisal_dataset_10000.csv"
):
    """
    Generate dataset with controlled depth distribution.

    depth_distribution example:
    {
        0: 2500,
        1: 2500,
        2: 2000,
        3: 1500,
        4: 1000,
        5: 500
    }
    """
    if depth_distribution is None:
        depth_distribution = {
            0: 2500,
            1: 2500,
            2: 2000,
            3: 1500,
            4: 1000,
            5: 500
        }

    if sum(depth_distribution.values()) != total_sentences:
        raise ValueError("Sum of depth_distribution must equal total_sentences")

    rows = []
    seen = set()

    for depth, count in depth_distribution.items():
        generated = 0
        attempts = 0

        while generated < count:
            sent = generate_sentence(depth)
            attempts += 1

            # avoid exact duplicates
            if sent not in seen:
                seen.add(sent)
                rows.append({
                    "sentence": sent,
                    "depth": depth,
                    "length_words": len(sent[:-1].split()),  # exclude period
                    "type": "synthetic_center_embedding"
                })
                generated += 1

            # safety check in case uniqueness becomes hard
            if attempts > count * 50:
                print(f"Warning: uniqueness getting harder at depth {depth}")
                break

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(df)} sentences to {output_csv}")
    print(df["depth"].value_counts().sort_index())
    print(df.head(10))

    return df

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    df = generate_dataset()