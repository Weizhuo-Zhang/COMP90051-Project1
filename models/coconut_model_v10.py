import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertModel

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA!")
else:
    device = torch.device('cpu')


class CoconutModel(nn.Module):
    def __init__(self,
                 input_size=768,
                 drop_out_rate=0.2,
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
        print("Extract Model V10 FeatureOnly")
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        return

    def create_params(self):
        self.feature_lstm = nn.LSTM(input_size=self.input_size,
                                    hidden_size=64,
                                    num_layers=1,
                                    batch_first=True)
        self.feature_layer = nn.Linear(64, self.feature_size)
        self.dropout = nn.Dropout(self.drop_out_rate)
        self.reset_params()
        return

    def reset_params(self):
        nn.init.xavier_normal_(self.feature_layer.weight)

    def prepare_data_for_coconut_model(self, sentences):
        sentences = list(sentences)
        max_length = 0
        tokenized_text_list = []
        for batch_sentences in sentences:
            tokenized_text = ['[CLS]'] + self.tokenizer.tokenize(batch_sentences.lower()) + ['[SEP]']
            attention_mask = [1] * len(tokenized_text)
            if len(tokenized_text) > 512:
                # TODO: Drop Long Sequence for now
                tokenized_text = tokenized_text[:512]
                attention_mask = attention_mask[:512]
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
        outputs = self.bert_model(tokens_tensor, token_type_ids=segments_tensor, attention_mask=attention_tensor)
        last_hidden_state, feature, hidden_states, attentions = outputs
        lstm_out, (last_hidden_state, last_cell_state) = self.feature_lstm(last_hidden_state)
        feature = self.feature_layer(last_hidden_state)
        feature = F.normalize(feature, dim=1, p=2)
        feature = feature.view(-1, 192)
        return feature
