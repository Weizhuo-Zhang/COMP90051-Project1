import torch
from models.coconut_model_v11 import CoconutModel


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
        self.model.bert_model.to(self.device)
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
        self.model.bert_model.eval()
        with torch.no_grad():
            features = self.model(sentences)
        if torch.cuda.is_available():
            features = features.cpu().detach().numpy()
            torch.cuda.empty_cache()
            return features
        return features.numpy()


if __name__ == '__main__':
    test_model = FeatureExtract(checkpoints_path='/Users/hanxunhuang/Desktop/checkpoints/margin1.0_v6_epoch1.pth')
    feature = test_model.get_features(["Ready for another killer Tap Tap Thursday? Get Medicate from now in Tap Tap Revenge - (iDevice only link!)",
                                       ])
    # feature1 = feature[0] # ID1
    # feature2 = feature[1] # ID2
    # feature3 = feature[2] # ID3
    # from numpy.linalg import norm
    # positive_dist = norm(feature1-feature2)
    # negative_dist = norm(feature1-feature3)
    # print("Negative", negative_dist)
    # print("Positive", positive_dist)
