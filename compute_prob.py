from sentence_transformers import SentenceTransformer


# Input file path
# test_tweets_unlabeled_dataframe_v1_7
train_data_path = "data/v7/train_set_v1_7_full.txt"
test_data_path = "data/v7/test_tweets_unlabeled_dataframe_v1_7.txt"

# Output file path
train_embed_out_path = 'data/v7/full_siamese_bert_base_cased_v7.0C_triplet_epoch1'
test_embed_out_path = 'data/v7/test_siamese_bert_base_cased_v7.0C_triplet_epoch1'
checkpoints_path = 'checkpoints/sentence_transformers/bert_base_cased_200_v7.0C_triplet_epoch1'


test_model = SentenceTransformer(checkpoints_path)

import pandas as pd
import numpy as np

def get_sentence_list(data_path, sentence_index=0):
    df_data = pd.read_csv(data_path, sep='\t', header=None)
    df_data = np.array(df_data)
    sentence_list = df_data[:, sentence_index]
    return sentence_list

def get_embed(sentences):
    return test_model.encode(sentences, batch_size=512)

# Read Data
train_sentence = list(get_sentence_list(train_data_path, sentence_index=1))
test_sentence = list(get_sentence_list(test_data_path, sentence_index=0))

print(len(train_sentence), type(train_sentence), type(train_sentence[0]))
print(len(test_sentence), type(test_sentence), type(test_sentence[0]))

test_feature_list = get_embed(test_sentence)
train_feature_list = get_embed(train_sentence)
print(len(test_feature_list), len(train_feature_list))



test_feature_list_np_array = np.asarray(test_feature_list)
train_feature_list_np_array = np.asarray(train_feature_list)

print(test_feature_list_np_array.shape)
print(train_feature_list_np_array.shape)

np.save(test_embed_out_path, test_feature_list_np_array)
np.save(train_embed_out_path, train_feature_list_np_array)
