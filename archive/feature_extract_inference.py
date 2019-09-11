import torch
from models.coconut_extract_v2 import CoconutFeatureExtract
from pytorch_transformers import BertTokenizer, BertModel
from coconut_train import prepare_data_for_coconut_model


class FeatureExtract:
    def __init__(self,
                 num_of_classes=1132,
                 feature_size=192,
                 checkpoints_path="checkpoints/coconut_extract_model_v2.pth"):
        self.num_of_classes = num_of_classes
        self.feature_size = feature_size
        self.checkpoints_path = checkpoints_path

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        self.bert_model.eval()

        self.model = CoconutFeatureExtract(num_of_classes=num_of_classes,
                                           feature_size=feature_size)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.bert_model = self.bert_model.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.load_model()
        return

    def load_model(self):
        checkpoints = torch.load(self.checkpoints_path, map_location=self.device)
        self.model.load_state_dict = checkpoints["model_state_dict"]
        return

    # Input list of sentence => Return nparry of features
    # Output Shape [BatchSize, FeatureDimensions]
    def get_features(self, sentences):
        self.bert_model.eval()
        self.model.eval()

        if len(sentences) > 1:
            self.model.batch(True)

        embeds, _ = prepare_data_for_coconut_model((sentences, None), self.bert_model, self.tokenizer)

        with torch.no_grad():
            features = self.model(embeds)[0]

        if torch.cuda.is_available():
            return features.cpu().detach().numpy()
        return features.numpy()


if __name__ == '__main__':
    test_model = FeatureExtract()
    feature = test_model.get_features(["Sibername Custom Web Site Design News Release", "thanks I appreciate the special consideration..... :)"])
    feature1 = feature[0]
    feature2 = feature[1]
    from numpy import dot
    from numpy.linalg import norm

    cos_sim = dot(feature1, feature2)/(norm(feature1)*norm(feature2))
    print(cos_sim)
