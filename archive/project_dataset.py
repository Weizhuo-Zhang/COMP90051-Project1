import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ProjectDataset(Dataset):
    def __init__(self, file_path, idx_file_path, use_group=False):
        with open(idx_file_path, 'rb') as handle:
            self.class_idx = pickle.load(handle)
            self.idx_class = dict((y, x) for x, y in self.class_idx.items())

        self.samples_frame = pd.read_csv(file_path, sep='\t', names=["ID", "Sentence"])
        self.labels = []
        self.sentences = []
        self.use_group = use_group
        return

    def __len__(self):
        return len(self.samples_frame)

    def __getitem__(self, idx):
        id, sentnece = self.samples_frame.iloc[idx, 0:].values
        sentnece = str(sentnece)
        if self.use_group:
            group_id, id_in_group = self.class_idx[id]
            group_id = torch.tensor(group_id)
            id_in_group = torch.tensor(id_in_group)
            return sentnece, group_id, id_in_group
        else:
            id_indice = self.class_idx[id]
            id_indice = torch.tensor(id_indice)
            return sentnece, id_indice
