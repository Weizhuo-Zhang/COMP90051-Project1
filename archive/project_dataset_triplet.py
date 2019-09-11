import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class ProjectDataset(Dataset):
    def __init__(self, file_path=None):
        self.samples_frame = pd.read_csv(file_path, sep='\t', names=["Sentence1", "Sentence2", "Labels"])
        self.labels = []
        self.sentences = []
        return

    def __len__(self):
        return len(self.samples_frame)

    def get_one_hot(self, id):
        indices = torch.tensor(self.class_idx[id])
        one_hot_id = torch.nn.functional.one_hot(indices, self.num_of_classes)
        return torch.FloatTensor(one_hot_id.unsqueeze(0))

    def __getitem__(self, idx):
        row = self.samples_frame.iloc[idx, 0:].values
        anchor = row[0]
        positive = row[1]
        negative = row[2]
        return anchor, positive, negative


if __name__ == "__main__":
    from pytorch_transformers import BertTokenizer, BertModel
    from tqdm import tqdm
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def prepare_data_for_coconut_model(sentences):
        sentences = list(sentences)
        tokens_tensor_batch = []
        segments_tensors_batch = []
        for batch_sentences in sentences:
            tokenized_text = ['[CLS]'] + tokenizer.tokenize(batch_sentences.lower()) + ['[SEP]']
            if len(tokenized_text) > 512:
                # TODO: Drop Long Sequence for now
                tokenized_text = tokenized_text[:512]
            segments_ids = [0] * len(tokenized_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor(indexed_tokens).to(device)
            segments_ids_tensor = torch.tensor(segments_ids).to(device)
            tokens_tensor_batch.append(tokens_tensor)
            segments_tensors_batch.append(segments_ids_tensor)
        segments_tensor = torch.nn.utils.rnn.pad_sequence(segments_tensors_batch, batch_first=True)
        tokens_tensor = torch.nn.utils.rnn.pad_sequence(tokens_tensor_batch, batch_first=True)
        return tokens_tensor, segments_tensor

    data_set = ProjectDataset(file_path="data/triplet/triple_sentences.csv")
    # train_indices = list(range(0, 1000000))
    # valid_indices = list(range(1000000, len(data_set)))
    # train_set = Subset(data_set, train_indices)
    # valid_set = Subset(data_set, valid_indices)

    # train_loader = DataLoader(train_set,
    #                           batch_size=1,
    #                           shuffle=False,
    #                           pin_memory=True,
    #                           num_workers=4)
    #
    # dev_loader = DataLoader(valid_set,
    #                         batch_size=1,
    #                         shuffle=False,
    #                         pin_memory=True,
    #                         num_workers=4)

    loader = DataLoader(data_set,
                        batch_size=256,
                        shuffle=False,
                        pin_memory=True,
                        num_workers=4)
    print(len(loader))

    for i, batch in tqdm(enumerate(loader)):
        anchor, positive, negative = batch
        anchor_token, anchor_segments = prepare_data_for_coconut_model(anchor)
        positive_token, positive_segments = prepare_data_for_coconut_model(positive)
        negative_token, negative_segments = prepare_data_for_coconut_model(negative)

        with torch.no_grad():
            _, anchor_feature, _, _ = bert_model(anchor_token, token_type_ids=anchor_segments)
            _, positive_feature, _, _ = bert_model(positive_token, token_type_ids=positive_segments)
            _, negative_feature, _, _ = bert_model(negative_token, token_type_ids=negative_segments)
            print(anchor_feature.shape)
            row = torch.cat([anchor_feature.unsqueeze(1), positive_feature.unsqueeze(1), negative_feature.unsqueeze(1)], dim=1)
            print(row.shape)
