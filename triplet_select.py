# Triplet Select Samples
# Anchor Positive Negative
import csv
import collections
import random
import pandas as pd
from tqdm import tqdm


def build_data_set(input_file_path, out_file_path, total_nums_of_rows=200000):
    print(input_file_path)
    train_dict = collections.defaultdict(list)
    with open(input_file_path, encoding='utf-8') as tsvfile:
        reader = tsvfile.readlines()
        for i, row in enumerate(reader):
            row = row.strip().split("\t")
            id = int(row[0])
            instance = row[1]
            train_dict[id].append(instance)
        print("Total rows: %d" % i)
    print("Total ids: %d" % len(train_dict))
    df = pd.DataFrame()
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

        dataset_list.append([anchor, positive, negative])

    df = pd.DataFrame((dataset_list))
    print(df.head())
    print(len(df))
    df.to_csv(out_file_path, index=False, sep='\t', header=None)
    print(out_file_path, ' Saved!')


build_data_set(input_file_path="data/v7/train_set_v1_7_full.txt",
               out_file_path="data/v7/triplet_train_full.csv",
               total_nums_of_rows=1000000)
#
# build_data_set(input_file_path="data/v7/dev_set_v1_7.txt",
#                out_file_path="data/v7/triplet_dev_200.csv",
#                total_nums_of_rows=200000)
