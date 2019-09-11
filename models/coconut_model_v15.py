import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA!")
else:
    device = torch.device('cpu')


class TrainableEltwiseLayer(nn.Module):
    def __init__(self, size):
        super(TrainableEltwiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(1, size))  # define the trainable parameter

    def forward(self, x):
        return x * self.weights


class CoconutModel(nn.Module):
    def __init__(self,
                 input_size=768,
                 drop_out_rate=0.1,
                 feature_size=192):
        super(CoconutModel, self).__init__()
        self.input_size = input_size
        self.drop_out_rate = drop_out_rate
        self.feature_size = feature_size
        self.create_params()
        print("Extract Model V15 FeatureOnly")
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        return

    def create_params(self):
        # self.fc1 = TrainableEltwiseLayer(768)
        self.fc1 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(self.drop_out_rate)
        self.reset_params()
        return

    def reset_params(self):
        # self.fc1.weights.data.fill_(1)
        nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, input_tensor):
        out = self.dropout(input_tensor)
        out = self.fc1(out)
        # out += input_tensor
        return out
