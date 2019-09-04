import torch
import torch.nn as nn
import pandas as pd
from project_dataset_prob import ProjectDataset
from torch.utils.data import DataLoader
from models.coconut_model_v13 import CoconutModel
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


dataset = ProjectDataset(file_path="data/pair_prob/k_top_pair_file_100_of_1000.csv")
data_loader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)

checkpoint_filename = 'checkpoints/v13_best_epoch2.pth'
checkpoints = torch.load(checkpoint_filename, map_location=device)
model = CoconutModel()
model.load_state_dict(checkpoints["model_state_dict"])
model.eval()
model.to(device)

df = pd.DataFrame()

softmax = nn.Softmax(dim=1)
for i, batch in tqdm(enumerate(data_loader)):
    sentence1, sentence2 = batch
    with torch.no_grad():
        pred = model((sentence1, sentence2))
    pred = softmax(pred).tolist()
    new_df = pd.DataFrame(pred)
    df = df.append(new_df)

export_csv = df.to_csv('data/pair_prob/prob_out_k_top_pair_file_100_of_1000.csv', sep='\t', index=None, header=None)
