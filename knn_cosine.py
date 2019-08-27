import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cosine

train_set_path = "./data/BERT_encoding/train_encode_tuned.npy"
dev_set_path = "./data/BERT_encoding/dev_encode_tuned.npy"
train_set_csv = "./data/train_set_v1.txt"
dev_set_csv = "./data/dev_set_v1.txt"

train_set = np.load(train_set_path)
train_csv = pd.read_csv(train_set_csv, sep='\t', header=None)
train_csv = np.array(train_csv)
dev_set = np.load(dev_set_path)
dev_csv = pd.read_csv(dev_set_csv, sep='\t', header=None)
dev_csv = np.array(dev_csv)

train_label = (train_csv[:,0]).astype('int')
dev_label = (dev_csv[:,0]).astype('int')


k = 13
num_predicting = 10
knn_clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=k, metric=cosine)
knn_clf.fit(train_set, train_label)
predicted = knn_clf.predict(dev_set[:num_predicting])
accuracy = sum(predicted == dev_label[:num_predicting])/len(predicted)
print(accuracy)