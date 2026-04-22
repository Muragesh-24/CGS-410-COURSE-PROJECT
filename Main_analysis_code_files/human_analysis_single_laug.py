"""
Linguistic Analysis Module for Dependency Structures

This module analyzes dependency length, tree height, and related linguistic 
properties from CoNLL-U formatted dependency treebank data.
"""

import matplotlib.pyplot as plt
import numpy as np
from conllu import parse
from scipy.stats import pearsonr
import os


def analyze_dependency_lengths(file_path):
    """
    Analyze dependency lengths in a CoNLL-U formatted file.
    
    Calculates overall average dependency length, maximum dependency length,
    per-sentence averages, and the full distribution of all dependency lengths.
    
    Args:
        file_path (str): Path to the CoNLL-U formatted file
        
    Returns:
        dict: Dictionary containing:
            - "overall_avg": Mean dependency length across all tokens
            - "max_dl": Maximum dependency length observed
            - "sentence_avg": List of average dependency lengths per sentence
            - "distribution": List of all individual dependency lengths
    """
    # Read and parse the CoNLL-U file
    with open(file_path, "r", encoding="utf-8") as f:
        sentences = parse(f.read())

    sentence_avg = []  # Average DL for each sentence
    all_lengths = []   # All individual dependency lengths
    max_dl = 0         # Maximum dependency length found

    # Process each sentence
    for sentence in sentences:
        total = 0      # Sum of dependency lengths in sentence
        deps = 0       # Count of dependencies in sentence

        # Process each token in the sentence
        for token in sentence:
            # Skip non-integer token IDs (such as multi-word tokens)
            if type(token["id"]) != int:
                continue

            head = token["head"]

            # Calculate dependency length if token has a dependency head
            if head != 0:
                # Dependency length is absolute distance between token and head
                dl = abs(token["id"] - head)
                all_lengths.append(dl)
                
                total += dl
                deps += 1

                # Track maximum dependency length
                if dl > max_dl:
                    max_dl = dl

        # Calculate and store average dependency length for this sentence
        if deps > 0:
            sentence_avg.append(total / deps)

    # Calculate overall average dependency length
    overall = sum(all_lengths) / len(all_lengths)

    return {
        "overall_avg": overall,
        "max_dl": max_dl,
        "sentence_avg": sentence_avg,
        "distribution": all_lengths
    }


# ============================================================================
# MAIN ANALYSIS: Dependency Length Statistics
# ============================================================================

file_path = "../data/en_ewt-ud-dev.conllu.txt"
english = analyze_dependency_lengths(file_path)

# Print basic statistics for dependency lengths
print("\n" + "="*60)
print("DEPENDENCY LENGTH STATISTICS")
print("="*60)

print(f"Overall Average DL: {english['overall_avg']:.4f}")
print(f"Maximum DL: {english['max_dl']}")
print(f"First 10 sentence averages:")
print([f"{x:.4f}" for x in english["sentence_avg"][:10]])
print(f"Standard Deviation: {np.std(english['distribution']):.4f}")
print(f"Median DL: {np.median(english['distribution']):.4f}")
print(f"90th percentile: {np.percentile(english['distribution'], 90):.4f}")
print(f"95th percentile: {np.percentile(english['distribution'], 95):.4f}")

# Visualize dependency length distribution
plt.figure(figsize=(10, 6))
plt.hist(english["distribution"], bins=30, edgecolor='black', alpha=0.7)
plt.title("Dependency Length Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Dependency Length", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Create boxplot for dependency lengths
plt.figure(figsize=(8, 6))
plt.boxplot(english["distribution"], vert=True)
plt.title("Dependency Length Boxplot", fontsize=14, fontweight='bold')
plt.ylabel("Dependency Length", fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()




# ============================================================================
# ANALYSIS: Sentence Length vs. Dependency Length Relationship
# ============================================================================

def sentence_length_vs_dl(file_path):
    """
    Calculate sentence length and average dependency length for each sentence.
    
    Analyzes the relationship between how long a sentence is and how complex
    its dependency structure is (as measured by average dependency length).
    
    Args:
        file_path (str): Path to the CoNLL-U formatted file
        
    Returns:
        tuple: (sentence_lengths, sentence_dl_avgs)
            - sentence_lengths: List of sentence lengths (number of tokens)
            - sentence_dl_avgs: List of average dependency lengths per sentence
    """
    # Read and parse the CoNLL-U file
    with open(file_path, encoding="utf-8") as f:
        sentences = parse(f.read())

    sentence_lengths = []  # Store length of each sentence
    sentence_dl = []       # Store average dependency length per sentence

    # Process each sentence
    for sentence in sentences:
        total = 0      # Sum of dependency lengths in sentence
        deps = 0       # Count of dependencies in sentence

        # Process each token in the sentence
        for token in sentence:
            # Skip non-integer token IDs (such as multi-word tokens)
            if type(token["id"]) != int:
                continue

            # Calculate dependency length if token has a dependency head
            if token["head"] != 0:
                total += abs(token["id"] - token["head"])
                deps += 1

        # Store sentence data if it has dependencies
        if deps > 0:
            sentence_lengths.append(len(sentence))
            sentence_dl.append(total / deps)

    return sentence_lengths, sentence_dl


# Calculate sentence length vs. dependency length data
x, y = sentence_length_vs_dl(file_path)

# Calculate and report Pearson correlation
corr, _ = pearsonr(x, y)
print(f"\nPearson Correlation (Sentence Length vs. Avg DL): {corr:.4f}")

# Visualize the relationship
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, s=30)
plt.xlabel("Sentence Length (number of tokens)", fontsize=12)
plt.ylabel("Average Dependency Length", fontsize=12)
plt.title("Sentence Length vs. Dependency Length", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# ============================================================================
# ANALYSIS: Tree Height Statistics
# ============================================================================

def tree_height(sentence):
    """
    Calculate the height (maximum depth) of a dependency tree.
    
    Determines the maximum distance from any node to the root (head=0)
    in the dependency tree structure of a sentence.
    
    Args:
        sentence (list): A sentence parsed from CoNLL-U format
        
    Returns:
        int: The maximum depth (height) of the dependency tree
    """
    # Create mapping of token ID to its head
    heads = {}

    for token in sentence:
        if type(token["id"]) == int:
            heads[token["id"]] = token["head"]

    def get_depth(node):
        """
        Calculate depth of a node (distance from root).
        
        Recursively traverses upward through heads until reaching root (0).
        
        Args:
            node (int): Token ID to find depth for
            
        Returns:
            int: Distance from node to root
        """
        depth = 0

        # Walk up the tree until reaching root
        while node != 0:
            node = heads.get(node, 0)
            depth += 1

        return depth

    # Find maximum depth across all nodes
    max_depth = 0

    for node in heads:
        d = get_depth(node)

        if d > max_depth:
            max_depth = d

    return max_depth


# Calculate tree heights for all sentences
heights = []

with open(file_path, "r", encoding="utf-8") as f:
    sentences = parse(f.read())

for sentence in sentences:
    h = tree_height(sentence)
    heights.append(h)

# Print tree height statistics
print("\n" + "="*60)
print("TREE HEIGHT STATISTICS")
print("="*60)

print(f"Average Tree Height: {np.mean(heights):.4f}")
print(f"Maximum Tree Height: {max(heights)}")
print(f"Median Tree Height: {np.median(heights):.4f}")
print(f"Min Tree Height: {min(heights)}")

# Visualize tree height distribution
plt.figure(figsize=(10, 6))
plt.hist(heights, bins=20, edgecolor='black', alpha=0.7)
plt.title("Tree Height Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Tree Height (maximum depth)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================================
# CORRELATION ANALYSIS: Tree Height vs. Average Dependency Length
# ============================================================================

# Calculate average dependency length for each sentence
avg_dl = []

for sentence in sentences:
    total = 0
    deps = 0

    for token in sentence:
        # Skip non-integer token IDs
        if type(token["id"]) != int:
            continue

        # Calculate dependency length if token has a head
        if token["head"] != 0:
            total += abs(token["id"] - token["head"])
            deps += 1

    # Store average DL if sentence has dependencies
    if deps > 0:
        avg_dl.append(total / deps)

# Calculate and report correlation
heights = np.array(heights)
avg_dl = np.array(avg_dl)

# Align lengths by truncating to the shorter array
min_len = min(len(heights), len(avg_dl))
heights_aligned = heights[:min_len]
avg_dl_aligned = avg_dl[:min_len]

# Pearson correlation
corr_tree_dl, _ = pearsonr(heights_aligned, avg_dl_aligned)
print(f"Pearson Correlation (Tree Height vs. Avg DL): {corr_tree_dl:.4f}")

# Scatter plot
plt.scatter(heights_aligned, avg_dl_aligned, alpha=0.5, s=30)
plt.xlabel("Tree Height")
plt.ylabel("Average Dependency Length")
plt.title("Tree Height vs. Avg DL")
plt.show()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

