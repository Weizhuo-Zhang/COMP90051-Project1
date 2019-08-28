import os
import time
import argparse
import pandas as pd
import numpy as np

def get_k_folders(X, k_folders=7):
    m = len(X)
    mini_batch_size = int(m/k_folders)
    mini_batches = []

    for k in range(k_folders-1):
        mini_batch_X = X[k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batches.append(mini_batch_X)

    mini_batch_X = X[(k_folders-1) * mini_batch_size:]
    mini_batches.append(mini_batch_X)

    return mini_batches

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _main(args):
    train_feature_set_path = args.feature_path
    train_set_path = args.dataset_file_path

    train_feature_set = np.load(train_feature_set_path)
    train_set = pd.read_csv(train_set_path, sep="\t", header=None)
    train_set_np = np.array(train_set)
    train_label = (train_set_np[:,0]).astype('int')

    now = time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime(time.time()))
    file_prefix = "data/split_{0}/split_".format(now)

    k_folders = 7
    feature_mini_batches = get_k_folders(train_feature_set, k_folders)
    label_mini_batches = get_k_folders(train_label, k_folders)
    for i, feature_batch in enumerate(feature_mini_batches):
        folder_path = "{0}{1}".format(file_prefix, i+1)
        create_dir(folder_path)
        label_batch = label_mini_batches[i]
        feature_batch_file_path = \
                "{0}/train_split_feature_{1}".format(folder_path, i+1)
        label_batch_file_path = \
                "{0}/train_split_label_{1}".format(folder_path, i+1)
        os.system("cp {0} {1}/train_feature.npy".format(train_feature_set_path, folder_path))
        os.system("cp {0} {1}/train_set.txt".format(train_set_path, folder_path))
        os.system("cp knn_split_top_k_candidate.py {1}".format(train_set_path, folder_path))
        np.save(feature_batch_file_path, feature_batch)
        np.save(label_batch_file_path, label_batch)

if "__main__" == __name__:
    parser = argparse.ArgumentParser(description="Split training set features")
    parser.add_argument(
        'feature_path', type=str,
        help = "the path of input feature 'npy' file")
    parser.add_argument(
        'dataset_file_path', type=str,
        help = "the path of training dataset file")

    args = parser.parse_args()
    _main(args)
