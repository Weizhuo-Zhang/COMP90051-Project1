import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, separable=False):
        super(ConvBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.separable = separable

        padding = (kernel_size - 1) // 2
        self.out_conv = nn.Sequential(nn.Conv2d(in_planes,
                                                out_planes,
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                bias=False),
                                      nn.BatchNorm2d(out_planes, track_running_stats=False),
                                      nn.ReLU())

    def forward(self, x):
        out = self.out_conv(x)
        return out


class CoconutModel(nn.Module):
    def __init__(self,
                 input_size=768,
                 drop_out_rate=0.4,
                 num_of_group=98,
                 num_of_classes=95,
                 feature_size=192,
                 num_attention_heads=12):
        super(CoconutModel, self).__init__()
        self.input_size = input_size
        self.drop_out_rate = drop_out_rate
        self.num_of_group = num_of_group
        self.num_of_classes = num_of_classes
        self.num_attention_heads = num_attention_heads
        self.feature_size = feature_size
        self.use_batch = False
        self.feature_size = feature_size
        self.create_params()
        print("Model-v3-cnn\tNumber of Groups: %d\tNumber of id in Group: %d" % (self.num_of_group, self.num_of_classes))

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        return

    def create_params(self):
        self.conv1 = ConvBlock(13, self.feature_size, 3)
        self.group_classfiy = nn.Linear(self.feature_size, self.num_of_group, bias=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
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
        concat_hiddens = None
        for item in hidden_states:
            if concat_hiddens is None:
                concat_hiddens = item[:, None, :, :]
            else:
                item = item[:, None, :, :]
                concat_hiddens = torch.cat([concat_hiddens, item], dim=1)
        concat_hiddens = torch.transpose(concat_hiddens, 2, 3)
        concat_hiddens = self.conv1(concat_hiddens)
        concat_hiddens = self.global_avg_pool(concat_hiddens)
        concat_hiddens = concat_hiddens.view(-1, self.feature_size)
        out = self.dropout(concat_hiddens)
        out = self.group_classfiy(out)
        return out
