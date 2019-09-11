import torch
from models.coconut_model_v13 import CoconutModel


class FeatureExtract:
    def __init__(self,
                 checkpoints_path="checkpoints/v13_best.pth"):

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.checkpoints_path = checkpoints_path
        self.model = CoconutModel()
        self.model.to(self.device)
        self.model.bert_model.to(self.device)
        self.load_model()
        return

    def load_model(self):
        checkpoints = torch.load(self.checkpoints_path, map_location=self.device)
        self.model.load_state_dict = checkpoints["model_state_dict"]
        return

    # Input list of sentence => Return nparry of features
    # Output Shape [BatchSize, FeatureDimensions]
    def get_features(self, sentences, feature_type='Mean'):
        self.model.eval()
        self.model.bert_model.eval()
        with torch.no_grad():
            features = self.model(sentences, is_pair=False, feature_type=feature_type)
        if torch.cuda.is_available():
            features = features.cpu().detach().numpy()
            torch.cuda.empty_cache()
            return features
        return features.numpy()
