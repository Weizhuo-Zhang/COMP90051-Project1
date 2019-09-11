import torch
import numpy as np
from torch.utils.data import Dataset

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ProjectDataset(Dataset):
    def __init__(self, anchor_file=None, positive_file=None, negative_file=None):
        self.anchor_list = np.load(anchor_file)
        self.positive_list = np.load(positive_file)
        self.negative_list = np.load(negative_file)
        return

    def __len__(self):
        return len(self.anchor_list)

    def __getitem__(self, idx):
        return self.anchor_list[idx], self.positive_list[idx], self.negative_list[idx]
        # return torch.tensor(self.anchor_list[idx].tolist()), torch.tensor(self.positive_list[idx].tolist()), torch.tensor(self.negative_list[idx].tolist())
