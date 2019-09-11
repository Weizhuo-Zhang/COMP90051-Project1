import torch
from models.coconut_model_v15 import CoconutModel
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class FeatureExtract:
    def __init__(self,
                 feature_size=192,
                 checkpoints_path="checkpoints/coconut_extract_model_v2.pth"):

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.feature_size = feature_size
        self.checkpoints_path = checkpoints_path
        self.model = CoconutModel(feature_size=feature_size)
        self.model.to(self.device)
        self.load_model()
        return

    def load_model(self):
        checkpoints = torch.load(self.checkpoints_path, map_location=self.device)
        self.model.load_state_dict = checkpoints["model_state_dict"]
        return

    # Input list of sentence => Return nparry of features
    # Output Shape [BatchSize, FeatureDimensions]
    def get_features(self, sentences):
        self.model.eval()
        sentences = torch.tensor(sentences).to(device)
        with torch.no_grad():
            features = self.model(sentences)
        if torch.cuda.is_available():
            features = features.cpu().detach().numpy()
            torch.cuda.empty_cache()
            return features
        return features.numpy()
