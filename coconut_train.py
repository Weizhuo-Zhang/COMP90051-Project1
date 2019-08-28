import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch_transformers import BertTokenizer, BertModel, BertForSequenceClassification
from models.coconut_model_v2 import CoconutModel
from models.coconut_extract_v2 import CoconutFeatureExtract
from project_dataset import ProjectDataset
from torch.utils.data import DataLoader
from utils.utils import AverageMeter
from center_loss import CenterLoss
from radam import RAdam

parser = argparse.ArgumentParser(description='COMP90051 Project1')
# Training
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--l2_reg', type=float, default=0.00025)
parser.add_argument('--num_of_classes', type=int, default=9292)
parser.add_argument('--epoch', type=int, default=310)
parser.add_argument('--grad_clip_bound', type=float, default=5.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--default_bert', action='store_true', default=False)
parser.add_argument('--features_extract', action='store_true', default=False)
parser.add_argument('--feature_size', type=int, default=192)
parser.add_argument('--center_loss_alpha', type=float, default=0.1)
parser.add_argument('--center_loss_lr', type=float, default=0.01)

# FilePath
parser.add_argument('--check_point_path', type=str, default="checkpoints/")
parser.add_argument('--model_version_string', type=str, default="coconut_model_v1")
parser.add_argument('--train_set_file_path', type=str, default="data/v1/train_set_v1.txt")
parser.add_argument('--dev_set_file_path', type=str, default="data/v1/dev_set_v1.txt")
parser.add_argument('--idx_file_path', type=str, default="data/v1/v1_idx.pickle")
parser.add_argument('--log_every', type=int, default=200)

args = parser.parse_known_args()[0]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


GLOBAL_STEP = 0
CURRENT_BEST = 0
CURRENT_ACC = 0


def get_data_loader():
    data_loaders = {}

    train_set = ProjectDataset(file_path=args.train_set_file_path,
                               idx_file_path=args.idx_file_path,
                               num_of_classes=args.num_of_classes)

    dev_set = ProjectDataset(file_path=args.dev_set_file_path,
                             idx_file_path=args.idx_file_path,
                             num_of_classes=args.num_of_classes)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              pin_memory=True,
                              num_workers=4)

    dev_loader = DataLoader(dev_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=4)

    data_loaders["train_loader"] = train_loader
    data_loaders["dev_loader"] = dev_loader
    return data_loaders


def save_mode(epoch, model, optimizer, center_loss=None):
    global GLOBAL_STEP, CURRENT_BEST, CURRENT_ACC
    filename = args.check_point_path + args.model_version_string + '.pth'
    center_loss_state_dict = None
    if center_loss is not None:
        center_loss_state_dict = center_loss.state_dict()
    payload = {
        "epoch": epoch + 1,
        "args": args,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "center_loss_state_dict": center_loss_state_dict,
        "global_step": GLOBAL_STEP,
        "CURRENT_BEST": CURRENT_BEST
    }
    torch.save(payload, filename)
    print('\n%s Saved! \n' % (filename))

    if CURRENT_ACC > CURRENT_BEST:
        CURRENT_BEST = CURRENT_ACC
        filename = args.check_point_path + args.model_version_string + '_best.pth'
        torch.save(payload, filename)
        print('\n%s Saved! \n' % (filename))

    return


def load_model(model, optimizer, center_loss=None):
    global GLOBAL_STEP, CURRENT_BEST
    filename = args.check_point_path + args.model_version_string + '.pth'
    checkpoints = torch.load(filename)
    epoch = checkpoints["epoch"]
    model.load_state_dict(checkpoints["model_state_dict"])
    optimizer.load_state_dict(checkpoints["optimizer_state_dict"])
    GLOBAL_STEP = checkpoints["global_step"]
    CURRENT_BEST = checkpoints["CURRENT_BEST"]
    if center_loss is not None:
        center_loss.load_state_dict(checkpoints["center_loss_state_dict"])
    print('%s Loaded!' % (filename))
    return epoch, model, optimizer, center_loss


def prepare_data_for_coconut_model(batch, bert_model, tokenizer):
    sentences, labels = batch
    sentences = list(sentences)
    tokens_tensor_batch = []
    segments_tensors_batch = []
    for item in sentences:
        tokenized_text = tokenizer.tokenize(item.lower())

        if len(tokenized_text) > 512:
            # TODO: Drop Long Sequence for now
            tokenized_text = tokenized_text[:512]

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(indexed_tokens)

        tokens_tensor = torch.tensor(indexed_tokens).to(device)
        segments_tensors = torch.tensor(segments_ids).to(device)

        tokens_tensor_batch.append(tokens_tensor)
        segments_tensors_batch.append(segments_tensors)

    tokens_tensor = torch.nn.utils.rnn.pad_sequence(tokens_tensor_batch, batch_first=True)
    segments_tensors = torch.nn.utils.rnn.pad_sequence(segments_tensors_batch, batch_first=True)

    if args.default_bert:
        return tokens_tensor, labels

    with torch.no_grad():
        bert_model.to(device)
        outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)

    if torch.cuda.is_available() and labels is not None:
        labels = labels.cuda()

    return outputs, labels


def eval_model(epoch, model, loader, bert_model, tokenizer):
    global CURRENT_ACC
    model.eval()
    model.to(device)
    if not args.default_bert:
        model.batch(True)
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    print('=' * 20 + "Model Eval" + '=' * 20)
    for i, batch in tqdm(enumerate(loader)):
        start = time.time()

        input, labels = prepare_data_for_coconut_model(batch, bert_model, tokenizer)

        with torch.no_grad():
            if args.default_bert:
                pred = model(input)[0]
            elif args.features_extract:
                pred = model(input)[1]
            else:
                pred = model(input)
            loss = nn.CrossEntropyLoss()(pred, labels)
            train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
            acc_meter.update(train_acc.item())
            loss_meter.update(loss.item())
            end = time.time()
            used_time = end - start

        if (i) % args.log_every == 0:
            display = 'epoch=' + str(epoch) + \
                      '\tloss=%.6f' % (loss_meter.val) + \
                      '\tloss_avg=%.6f' % (loss_meter.avg) + \
                      '\tacc=%.4f' % (acc_meter.val) + \
                      '\tacc_avg=%.4f' % (acc_meter.avg) + \
                      '\ttime=%.2fit/s' % (1. / used_time)
            tqdm.write(display)
    print("Final Eval Acc: %.4f\n" % (acc_meter.avg))
    CURRENT_ACC = acc_meter.avg
    return


def train_model(epoch, model, optimizer, loader, bert_model, tokenizer, center_loss=None):
    global GLOBAL_STEP

    train_acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    c_loss_meter = AverageMeter()
    model.train()
    model.to(device)
    if not args.default_bert:
        model.batch(True)

    print('=' * 20 + "Model Training" + '=' * 20)

    for i, batch in tqdm(enumerate(loader)):
        start = time.time()

        input, labels = prepare_data_for_coconut_model(batch, bert_model, tokenizer)
        optimizer.zero_grad()
        model.zero_grad()

        if args.default_bert:
            pred = model(input)[0]
            loss = nn.CrossEntropyLoss()(pred, labels)
            loss.backward()
        elif args.features_extract:
            feature, pred = model(input)
            ce_loss = nn.CrossEntropyLoss()(pred, labels)
            c_loss = center_loss(feature, labels) * args.center_loss_alpha
            loss = ce_loss + c_loss
            loss.backward()
            for param in center_loss.parameters():
                param.grad.data *= (args.center_loss_lr / (args.center_loss_alpha * args.lr))
            c_loss_meter.update(c_loss.item())
        else:
            pred = model(input)
            loss = nn.CrossEntropyLoss()(pred, labels)
            loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_bound)
        optimizer.step()

        train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
        train_acc_meter.update(train_acc.item())
        loss_meter.update(loss.item())
        end = time.time()
        used_time = end - start

        if (GLOBAL_STEP) % args.log_every == 0:
            lr = optimizer.param_groups[0]['lr']
            display = 'epoch=' + str(epoch) + \
                      '\tglobal_step=%d' % (GLOBAL_STEP) + \
                      '\tloss=%.6f' % (loss_meter.val) + \
                      '\tloss_avg=%.6f' % (loss_meter.avg)
            if args.features_extract:
                display = display + '\tc_loss_avg=%.6f' % (c_loss_meter.avg)
            display = display + '\tlr=%.6f' % (lr) + \
                      '\tacc=%.4f' % (train_acc_meter.val) + \
                      '\tacc_avg=%.4f' % (train_acc_meter.avg) + \
                      '\ttime=%.2fit/s' % (1. / used_time)
            tqdm.write(display)
        GLOBAL_STEP += 1

    return


def train():
    # Init Training
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
    bert_model.eval()
    center_loss = None

    data_loaders = get_data_loader()
    if args.default_bert:
        coconut_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args.num_of_classes)
        params = list(coconut_model.parameters())
    elif args.features_extract:
        coconut_model = CoconutFeatureExtract(num_of_classes=args.num_of_classes,
                                              feature_size=args.feature_size)
        center_loss = CenterLoss(num_classes=args.num_of_classes,
                                 feat_dim=args.feature_size,
                                 use_gpu=torch.cuda.is_available())
        params = list(coconut_model.parameters()) + list(center_loss.parameters())
    else:
        coconut_model = CoconutModel()
        params = list(coconut_model.parameters())

    if torch.cuda.is_available():
        coconut_model = coconut_model.cuda()
        bert_model = bert_model.cuda()

    optimizer = RAdam(params=params,
                      lr=args.lr,
                      betas=(0.0, 0.999),
                      eps=1e-3,
                      weight_decay=args.l2_reg)
    starting_epoch = 0

    if args.resume:
        checkpoints = load_model(model=coconut_model,
                                 optimizer=optimizer,
                                 center_loss=center_loss)
        (starting_epoch, coconut_model, optimizer, center_loss) = checkpoints

    for epoch in range(starting_epoch, args.epoch):
        train_model(epoch=epoch,
                    model=coconut_model,
                    optimizer=optimizer,
                    loader=data_loaders["train_loader"],
                    tokenizer=tokenizer,
                    bert_model=bert_model,
                    center_loss=center_loss)
        eval_model(epoch=epoch,
                   model=coconut_model,
                   loader=data_loaders["dev_loader"],
                   tokenizer=tokenizer,
                   bert_model=bert_model)
        save_mode(epoch=epoch,
                  model=coconut_model,
                  optimizer=optimizer,
                  center_loss=center_loss)
    return


if __name__ == "__main__":
    train()
