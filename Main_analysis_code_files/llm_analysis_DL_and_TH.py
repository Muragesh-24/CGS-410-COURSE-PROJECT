# using our data for prediction
# making two types of sentence for prediction
# in prediction make use of other parameters too
# llm text dependence test
# What LLM does similar to humans
# What LLM does differently
# Why transformers might show DLM
# Limitations of study
#######################-------------------------------------------hand made dataset----------------------------#######################
# sentences = []

# with open("data/llm_generated.txt","r",encoding="utf8") as f:

#     current = ""

#     for line in f:

#         line = line.strip()

#         if not line:
#             continue

#         current += " " + line

#         if line.endswith("."):
#             sentences.append(current.strip())
#             current = ""


# if current:
#     sentences.append(current.strip())

# print("Total sentences:",len(sentences))

##############################-------------------------------------------dataset from kaggle----------------------------############################
import pandas as pd

df = pd.read_csv("../data/AIGTxt_dataset.csv", encoding='latin1')  # or 'ISO-8859-1'

sentences = df["ChatGPT-Generated"].tolist()  
print("Total sentences:", len(sentences))

import spacy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load model
nlp = spacy.load("en_core_web_sm")

# STORAGE LISTS
avg_dl = []
max_dl = []
tree_heights = []
sent_lengths = []

####################################################
# Dependency Length Function
####################################################

def dependency_metrics(doc):

    total_dl = 0
    dep_count = 0
    max_dep = 0

    for token in doc:

        if token.head != token:

            dl = abs(token.i - token.head.i)

            total_dl += dl
            dep_count += 1

            if dl > max_dep:
                max_dep = dl

    if dep_count == 0:
        return 0,0

    avg = total_dl / dep_count

    return avg,max_dep


####################################################
# Tree Height Function
####################################################

def get_tree_height(token):

    children = list(token.children)

    if not children:
        return 1

    return 1 + max(get_tree_height(child) for child in children)


def sentence_tree_height(doc):

    root = None

    for token in doc:

        if token.head == token:
            root = token
            break

    return get_tree_height(root)


####################################################
# MAIN ANALYSIS
####################################################
i=0
for sentence in sentences:

    if pd.isna(sentence):  # skip if the value is NaN
        continue
    sentence = str(sentence)  # ensure it's a string
    doc = nlp(sentence)
    print(i)
    i+=1

    avg,maxdep = dependency_metrics(doc)

    avg_dl.append(avg)

    max_dl.append(maxdep)

    tree_heights.append(sentence_tree_height(doc))

    sent_lengths.append(len(doc))


####################################################
# BASIC STATISTICS
####################################################

print("\n========== LLM STRUCTURAL ANALYSIS ==========")

print("Total sentences:",len(sentences))

print("\nSentence statistics")

print("Average sentence length:",np.mean(sent_lengths))

print("Maximum sentence length:",np.max(sent_lengths))

print("\nDependency statistics")

print("Average dependency length:",np.mean(avg_dl))

print("Maximum dependency length:",np.max(max_dl))

print("\nTree statistics")

print("Average tree height:",np.mean(tree_heights))

print("Maximum tree height:",np.max(tree_heights))


####################################################
# CORRELATION ANALYSIS
####################################################

corr_len_dl,_ = pearsonr(sent_lengths,avg_dl)

corr_height_dl,_ = pearsonr(tree_heights,avg_dl)

print("\nCorrelation analysis")

print("Sentence Length vs Avg DL:",corr_len_dl)

print("Tree Height vs Avg DL:",corr_height_dl)


####################################################
# GRAPHS
####################################################

# Graph 1
plt.figure(figsize=(6,5))

plt.scatter(sent_lengths,avg_dl,alpha=0.5)

plt.xlabel("Sentence Length")

plt.ylabel("Average Dependency Length")

plt.title("LLM Sentence Length vs Dependency Length")

plt.show()


# Graph 2
plt.figure(figsize=(6,5))

plt.scatter(tree_heights,avg_dl,alpha=0.5)

plt.xlabel("Tree Height")

plt.ylabel("Average Dependency Length")

plt.title("LLM Tree Height vs Dependency Length")

plt.show()


# Graph 3
plt.figure(figsize=(6,5))

plt.hist(avg_dl,bins=30)

plt.xlabel("Average Dependency Length")

plt.ylabel("Frequency")

plt.title("LLM Dependency Length Distribution")

plt.show()


####################################################
# SAMPLE OUTPUT CHECK
####################################################

print("\nSample sentences check:\n")

for i in range(5):

    print("Sentence:",sentences[i])

    print("Length:",sent_lengths[i])

    print("Avg DL:",avg_dl[i])

    print("Tree height:",tree_heights[i])

    print()