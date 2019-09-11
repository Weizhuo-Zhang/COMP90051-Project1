import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertModel

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA!")
else:
    device = torch.device('cpu')


class CoconutModel(nn.Module):
    def __init__(self,
                 input_size=768,
                 drop_out_rate=0.1,
                 feature_size=192,
                 num_attention_heads=12):
        super(CoconutModel, self).__init__()
        self.input_size = input_size
        self.drop_out_rate = drop_out_rate
        self.num_attention_heads = num_attention_heads
        self.feature_size = feature_size
        self.bert_model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True, output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        self.create_params()
        print("Extract Model V16 FeatureOnly")
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        return

    def create_params(self):
        self.feature_layer = nn.Sequential(nn.Linear(self.input_size, 2))
        self.dropout = nn.Dropout(self.drop_out_rate)
        self.reset_params()
        return

    def reset_params(self):
        for m in self.feature_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def prepare_data_for_coconut_model(self, sentences):
        sentences = list(sentences)
        max_length = 0
        tokenized_text_list = []
        for batch_sentences in sentences:
            tokenized_text = ['[CLS]'] + self.tokenizer.tokenize(batch_sentences.lower()) + ['[SEP]']
            attention_mask = [1] * len(tokenized_text)
            if len(tokenized_text) > 30:
                # TODO: Drop Long Sequence for now
                tokenized_text = tokenized_text[:30]
                attention_mask = attention_mask[:30]
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

    def prepare_data_for_coconut_model_pair(self, sentence1, sentence2):
        sentence1, sentence2 = list(sentence1), list(sentence2)
        max_length = 0
        tokenized_text_list = []
        sent1_lens = []
        sent2_lens = []
        for sent1, sent2 in zip(sentence1, sentence2):
            sent1 = self.tokenizer.tokenize(sent1)
            sent2 = self.tokenizer.tokenize(sent2)
            if len(sent1) > 30:
                sent1 = sent1[:30]
            if len(sent2) > 30:
                sent2 = sent2[:30]
            sent1 = ['[CLS]'] + sent1 + ['[SEP]']
            sent2 = sent2 + ['[SEP]']
            tokenized_text = sent1 + sent2
            sent1_lens.append(len(sent1))
            sent2_lens.append(len(sent2))
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

            segments_ids = [0] * sent1_lens[i] + [1] * (sent2_lens[i] + diff_length)
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

    def forward(self, sentence, is_pair=True, get_features=False):
        if is_pair:
            sentence1, sentence2 = sentence
            tokens_tensor, segments_tensor, attention_tensor = self.prepare_data_for_coconut_model_pair(sentence1, sentence2)
        else:
            tokens_tensor, segments_tensor, attention_tensor = self.prepare_data_for_coconut_model(sentence)
        outputs = self.bert_model(tokens_tensor, token_type_ids=segments_tensor, attention_mask=attention_tensor)
        last_hidden_state, cls_feature, hidden_states, attentions = outputs
        cls_feature = self.dropout(cls_feature)
        out = self.feature_layer(cls_feature)
        if get_features:
            return cls_feature
        return out
