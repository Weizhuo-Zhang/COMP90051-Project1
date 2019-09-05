import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

k = 101
split_num = os.getcwd().split('/')[-1].split("split_")[-1]

train_feature_path = "train_feature.npy"
train_set_path = "train_set.txt"
split_feature_path = "train_split_feature_{0}.npy".format(split_num)
split_label_path = "train_split_label_{0}.npy".format(split_num)

train_feature = np.load(train_feature_path)
train_set = pd.read_csv(train_set_path, sep='\t', header=None)
train_set = np.array(train_set)
train_sentence = train_set[:,1]
train_label = (train_set[:,0]).astype('int')
print("[SUCCESS] Loaded data")

split_feature = np.load(split_feature_path)
split_label = np.load(split_label_path)

print("[INFO] Fit knn model")
knn_clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=k, weights="distance")
knn_clf.fit(train_feature, train_label)
print("[SUCCESS] Successfully fitted knn model")

print("[INFO] Finding kneighbors")
k_neighbors_list = knn_clf.kneighbors(X=split_feature, return_distance=False)
print("[SUCCESS] Successfully Finded kneighbors")

pair_file_path = "split_pair_{0}.txt".format(split_num)
true_index_list = k_neighbors_list[:,0].repeat(k-1)
true_label_list = train_label[true_index_list]
true_sentence_list = train_sentence[true_index_list]

candidate_index_list = k_neighbors_list[:,1:].ravel()
candidate_label_list = train_label[candidate_index_list]
candidate_label_true_false = (candidate_label_list == true_label_list).astype('int')
candidate_sentence_list = train_sentence[candidate_index_list]
pair_df = pd.DataFrame({'true_sentence':true_sentence_list,
                        'candidate_sentence':candidate_sentence_list,
                        'pair_result':candidate_label_true_false})
pair_df.to_csv(pair_file_path, sep='\t', index=False)
