import torch
import numpy as np
from project_dataset_feature import ProjectDataset
from torch.utils.data import DataLoader
from models.coconut_model_v13 import CoconutModel
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def build_features(input_file, output_file):
    dataset = ProjectDataset(file_path=input_file)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)
    checkpoint_filename = 'checkpoints/v13_best.pth'
    checkpoints = torch.load(checkpoint_filename, map_location=device)
    model = CoconutModel()
    model.load_state_dict(checkpoints["model_state_dict"])
    model.eval()
    model.to(device)
    feature_list = None
    for i, batch in tqdm(enumerate(data_loader)):
        sentence = batch
        with torch.no_grad():
            pred = model(sentence, is_pair=False, feature_type='Mean')

        if torch.cuda.is_available():
            pred = pred.cpu().detach().numpy()
        else:
            pred = pred.numpy()
        if feature_list is None:
            feature_list = pred
        else:
            feature_list = np.concatenate((feature_list, pred), axis=0)

    np.save(output_file, feature_list)


if __name__ == '__main__':
    build_features('data/v5/dev_set_v1_5.txt', 'data/v5/v13_best_v5_dev')
    build_features('data/v5/train_set_v1_5.txt', 'data/v5/v13_best_v5_train')
