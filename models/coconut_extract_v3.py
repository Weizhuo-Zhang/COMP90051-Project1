import torch
import torch.nn as nn


class CoconutFeatureExtract(nn.Module):
    def __init__(self,
                 input_size=768,
                 drop_out_rate=0.2,
                 num_of_classes=9292,
                 feature_size=192,
                 num_attention_heads=12):
        super(CoconutFeatureExtract, self).__init__()
        self.input_size = input_size
        self.drop_out_rate = drop_out_rate
        self.num_of_classes = num_of_classes
        self.num_attention_heads = num_attention_heads
        self.feature_size = feature_size
        self.use_batch = False
        self.create_params()
        print("FeatureExtract-v3")
        return

    def create_params(self):
        self.layer1 = nn.Sequential(nn.Linear(self.input_size, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(512, 256),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU())
        self.atten_conv1 = nn.Sequential(nn.Conv2d(self.num_attention_heads * self.num_attention_heads,
                                                   self.num_attention_heads,
                                                   kernel_size=3,
                                                   padding=1,
                                                   stride=1,
                                                   bias=False),
                                         nn.BatchNorm2d(self.num_attention_heads),
                                         nn.ReLU())
        self.hidden_conv1 = nn.Sequential(nn.Conv2d(13,
                                                    self.num_attention_heads,
                                                    kernel_size=3,
                                                    padding=1,
                                                    stride=1,
                                                    bias=False),
                                          nn.BatchNorm2d(self.num_attention_heads),
                                          nn.ReLU())
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.features = nn.Linear(256 + 2 * self.num_attention_heads, self.feature_size)
        self.classfiy = nn.Linear(self.feature_size, self.num_of_classes, bias=False)
        self.dropout = nn.Dropout(self.drop_out_rate)
        self.reset_params()
        return

    def batch(self, use_batch):
        return

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, input_tensor):
        last_hidden_state, pooled_output, hidden_states, attentions = input_tensor

        # ConvOps on attentions heads
        concat_attentions = None
        for item in attentions:
            if concat_attentions is None:
                concat_attentions = item
            else:
                concat_attentions = torch.cat([concat_attentions, item], dim=1)
        concat_attentions = self.atten_conv1(concat_attentions)
        concat_attentions = self.global_avg_pool(concat_attentions)
        concat_attentions = concat_attentions.view(-1, self.num_attention_heads)

        # ConvOps on HiddenStates
        concat_hiddens = None
        for item in hidden_states:
            if concat_hiddens is None:
                concat_hiddens = item[:, None, :, :]
            else:
                item = item[:, None, :, :]
                concat_hiddens = torch.cat([concat_hiddens, item], dim=1)
        concat_hiddens = self.hidden_conv1(concat_hiddens)
        concat_hiddens = self.global_avg_pool(concat_hiddens)
        concat_hiddens = concat_hiddens.view(-1, self.num_attention_heads)

        # Classification Token Ops
        out = self.layer1(pooled_output)
        out = self.layer2(out)

        # Concat Features
        out = torch.cat([out, concat_attentions, concat_hiddens], dim=1)

        # FeatureExtract and classfiy
        features = self.features(out)
        out = self.dropout(features)
        out = self.classfiy(features)

        return features, out
