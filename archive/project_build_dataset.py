import argparse
import csv
import pickle
import torch
import os
from tqdm import tqdm
from pytorch_transformers import BertTokenizer, BertModel


parser = argparse.ArgumentParser(description='COMP90051 Project1')
parser.add_argument('--idx_file_path', type=str, default="data/v1/v1_idx.pickle")
parser.add_argument('--train_set_path', type=str, default="data/v1/train_set_v1.txt")
parser.add_argument('--dev_set_path', type=str, default="data/v1/dev_set_v1.txt")
parser.add_argument('--train_data_loaders_file_name', type=str, default="data/v1/train_dataloader")
parser.add_argument('--dev_data_loaders_file_name', type=str, default="data/v1/dev_dataloader")
parser.add_argument('--class_map_file_path', type=str, default="data/v1/class_map.pickle")
parser.add_argument('--num_of_classes', type=int, default=9292)
args = parser.parse_args()


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_embed(bert_model, indexed_tokens, segments_ids):
    if torch.cuda.is_available():
        bert_model = bert_model.cuda()
        tokens_tensor = torch.tensor(indexed_tokens).cuda()
        segments_tensors = torch.tensor(segments_ids).cuda()
    else:
        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor(segments_ids)

    with torch.no_grad():
        outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
    return outputs[0]


def get_bert_embed(bert_model, tokenizer, sentence):
    tokenized_text = tokenizer.tokenize(sentence)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(indexed_tokens)

    if len(indexed_tokens) > 512:
        indexed_tokens = list(chunks(indexed_tokens, 512))
        segments_ids = list(chunks(segments_ids, 512))
    else:
        indexed_tokens = [indexed_tokens]
        segments_ids = [segments_ids]

    out_tensor = None
    for i in range(len(indexed_tokens)):
        out = get_embed(bert_model, [indexed_tokens[i]], [segments_ids[i]])
        if out_tensor is None:
            out_tensor = out
        else:
            out_tensor = torch.cat([out_tensor, out], dim=1)
    return out_tensor


def build_file_loader(file_path, save_file_path, bert_model, tokenizer):
    global CLASS_IDX, IDX_CLASS

    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    ids = []
    sentences = []
    with open(file_path, encoding='utf-8') as file:
        reader = csv.reader((x.replace('\0', '') for x in file), delimiter='\t')
        for i, row in enumerate(reader):
            id = int(row[0])
            instance = row[1]
            ids.append(id)
            sentences.append(instance)

    # Build One Hot Labels
    payloads = []
    payload_batch_counter = 0
    for i in tqdm(range(len(sentences))):
        sentence = sentences[i]
        id = ids[i]
        if torch.cuda.is_available():
            indices = torch.tensor(CLASS_IDX[id]).cuda()
            one_hot = torch.nn.functional.one_hot(indices, args.num_of_classes)
        else:
            indices = torch.tensor(CLASS_IDX[id])
            one_hot = torch.nn.functional.one_hot(indices, args.num_of_classes)
        embed = get_bert_embed(bert_model, tokenizer, sentence)
        if embed.size(0) > 1 or embed.size(2) != 768:
            raise("embed error")
        payload = (embed.tolist(), one_hot.tolist())
        payloads.append(payload)
        if len(payloads) > 10000:
            file_name = save_file_path + '/dataloader_' + str(payload_batch_counter) + '.pickle'
            with open(file_name, 'wb') as file:
                pickle.dump(payloads, file)
                payloads = []
                payload_batch_counter += 1


def data_loader(bert_model, tokenizer):
    global CLASS_IDX, IDX_CLASS
    with open(args.idx_file_path, 'rb') as handle:
        CLASS_IDX = pickle.load(handle)
        IDX_CLASS = dict((y, x) for x, y in CLASS_IDX.items())
    build_file_loader(args.train_set_path, args.train_data_loaders_file_name, bert_model, tokenizer)
    build_file_loader(args.dev_set_path, args.dev_data_loaders_file_name, bert_model, tokenizer)
    payload = {}
    payload["CLASS_IDX"] = CLASS_IDX
    payload["IDX_CLASS"] = IDX_CLASS
    with open(args.class_map_file_path, 'wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.eval()
    data_loaders = data_loader(bert_model, tokenizer)
