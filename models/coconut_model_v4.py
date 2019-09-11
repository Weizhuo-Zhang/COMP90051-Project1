import torch
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertModel

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CoconutModel(nn.Module):
    def __init__(self,
                 input_size=768,
                 drop_out_rate=0.4,
                 num_of_classes=95,
                 feature_size=192,
                 num_attention_heads=12):
        super(CoconutModel, self).__init__()
        self.input_size = input_size
        self.drop_out_rate = drop_out_rate
        self.num_of_classes = num_of_classes
        self.num_attention_heads = num_attention_heads
        self.feature_size = feature_size
        self.use_batch = False
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.create_params()
        print("Extract Model V4 Classes: %d" % (self.num_of_classes))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        return

    def create_params(self):
        self.feature_layer = nn.Linear(self.input_size, self.feature_size)
        self.group_classfiy = nn.Linear(self.feature_size, self.num_of_classes, bias=False)
        self.dropout = nn.Dropout(self.drop_out_rate)
        self.reset_params()
        return

    def reset_params(self):
        nn.init.xavier_normal_(self.group_classfiy.weight)
        nn.init.xavier_normal_(self.feature_layer.weight)

    def prepare_data_for_coconut_model(self, sentences):
        sentences = list(sentences)
        tokens_tensor_batch = []
        segments_tensors_batch = []
        for batch_sentences in sentences:
            tokenized_text = ['[CLS]'] + self.tokenizer.tokenize(batch_sentences.lower()) + ['[SEP]']
            if len(tokenized_text) > 512:
                # TODO: Drop Long Sequence for now
                tokenized_text = tokenized_text[:512]
            segments_ids = [0] * len(tokenized_text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor(indexed_tokens).to(device)
            segments_ids_tensor = torch.tensor(segments_ids).to(device)
            tokens_tensor_batch.append(tokens_tensor)
            segments_tensors_batch.append(segments_ids_tensor)
        segments_tensor = torch.nn.utils.rnn.pad_sequence(segments_tensors_batch, batch_first=True)
        tokens_tensor = torch.nn.utils.rnn.pad_sequence(tokens_tensor_batch, batch_first=True)
        return tokens_tensor, segments_tensor

    def forward(self, sentences):
        tokens_tensor, segments_tensor = self.prepare_data_for_coconut_model(sentences)
        outputs = self.bert_model(tokens_tensor, token_type_ids=segments_tensor)
        last_hidden_state, pooled_output, hidden_states, attentions = outputs
        feature = self.feature_layer(pooled_output)
        out = self.dropout(feature)
        out = self.group_classfiy(out)
        return feature, out
