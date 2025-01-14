import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(ConvBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size

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
                 feature_size=192,
                 num_attention_heads=12):
        super(CoconutModel, self).__init__()
        self.input_size = input_size
        self.drop_out_rate = drop_out_rate
        self.num_attention_heads = num_attention_heads
        self.feature_size = feature_size
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.create_params()
        print("Extract Model V8 FeatureOnly")
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        return

    def create_params(self):
        self.conv1 = ConvBlock(in_planes=13, out_planes=64, kernel_size=3)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_layer = nn.Linear(64, 192)
        self.dropout = nn.Dropout(self.drop_out_rate)
        self.reset_params()
        return

    def reset_params(self):
        for m in self.conv1.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_normal_(self.feature_layer.weight)

    def prepare_data_for_coconut_model(self, sentences):
        sentences = list(sentences)
        max_length = 0
        tokenized_text_list = []
        for batch_sentences in sentences:
            tokenized_text = ['[CLS]'] + self.tokenizer.tokenize(batch_sentences.lower()) + ['[SEP]']
            if len(tokenized_text) > 512:
                # TODO: Drop Long Sequence for now
                tokenized_text = tokenized_text[:512]
            max_length = max(max_length, len(tokenized_text))
            tokenized_text_list.append(tokenized_text)

        indexed_tokens_batch = []
        segments_ids_token_batch = []
        attention_mask_token_batch = []
        for i, item in enumerate(tokenized_text_list):
            item_length = len(item)
            diff_length = max_length - item_length
            if diff_length > 0:
                item = item + ['[PAD]'] * diff_length

            segments_ids = [0] * item_length + [1] * diff_length
            attention_mask = [1] * item_length + [0] * diff_length
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(item)
            indexed_tokens_tensor = torch.tensor(indexed_tokens)
            segments_ids_tensor = torch.tensor(segments_ids)
            attention_mask_tensor = torch.tensor(attention_mask)

            indexed_tokens_batch.append(indexed_tokens_tensor)
            segments_ids_token_batch.append(segments_ids_tensor)
            attention_mask_token_batch.append(attention_mask_tensor)

        tokens_tensor = torch.stack(indexed_tokens_batch).to(device)
        segments_tensor = torch.stack(segments_ids_token_batch).to(device)
        attention_tensor = torch.stack(attention_mask_token_batch).to(device)

        return tokens_tensor, segments_tensor, attention_tensor

    def forward(self, sentences):
        tokens_tensor, segments_tensor, attention_tensor = self.prepare_data_for_coconut_model(sentences)
        with torch.no_grad():
            outputs = self.bert_model(tokens_tensor, token_type_ids=segments_tensor, attention_mask=attention_tensor)
        last_hidden_state, feature, hidden_states, attentions = outputs
        input = torch.transpose(torch.stack(list(hidden_states)), 0, 1)
        input = torch.transpose(input, 2, 3)
        input = self.conv1(input)
        input = self.global_avg_pool(input)
        input = input.view(-1, 64)
        feature = self.feature_layer(input)
        return feature
