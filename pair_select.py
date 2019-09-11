# Triplet Select Samples
# Anchor Positive Negative
import csv
import collections
import numpy as np

# Input File Path
train_file_path = "data/v7/dev_set_v1_7.txt"
# Save to File
file_path = 'data/pair_v7/paired_sentences_dev_fixed.csv'

train_dict = collections.defaultdict(list)

with open(train_file_path, encoding='utf-8') as tsvfile:
    reader = tsvfile.readlines()
    for i, row in enumerate(reader):
        row = row.strip().split("\t")
        id = int(row[0])
        instance = row[1]
        train_dict[id].append(instance)
    print("Total rows: %d" % i)
print("Total ids: %d" % len(train_dict))

# For Now, Just Use Random
import random
import pandas as pd
from tqdm import tqdm
#
df = pd.DataFrame()
total_nums_of_rows = 100000

dataset_list = []
for i in tqdm(range(total_nums_of_rows)):
    positive_id = random.choice(list(train_dict.keys()))
    negative_id = random.choice(list(train_dict.keys()))
    positive_sentence = train_dict[positive_id]
    negative_sentence = train_dict[negative_id]

    anchor = random.choice(positive_sentence)
    positive = random.choice(positive_sentence)
    negative = random.choice(negative_sentence)

    if anchor is None or positive is None or negative is None:
        raise('Error')

    dataset_list.append([anchor, positive, 1])
    dataset_list.append([anchor, negative, 0])

df = pd.DataFrame((dataset_list))
print(df.head())
print(len(df))
df.to_csv(file_path, index=False, sep='\t', header=None)
