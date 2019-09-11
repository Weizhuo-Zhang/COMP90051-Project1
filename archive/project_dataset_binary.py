import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ProjectDataset(Dataset):
    def __init__(self, file_path=None):
        self.samples_frame = pd.read_csv(file_path, sep='\t')
        return

    def __len__(self):
        return len(self.samples_frame)

    def __getitem__(self, idx):
        row = self.samples_frame.iloc[idx, 0:].values
        sentence1 = row[0]
        sentence2 = row[1]
        labels = row[2]
        return sentence1, sentence2, labels
