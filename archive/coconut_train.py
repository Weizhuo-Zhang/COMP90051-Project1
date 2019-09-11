import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch_transformers import BertTokenizer, BertModel, BertForSequenceClassification
from models.coconut_model_v4 import CoconutModel
from project_dataset import ProjectDataset
from torch.utils.data import DataLoader
from utils.utils import AverageMeter
from center_loss import CenterLoss
from radam import RAdam
from HierarchicalSoftmax import HierarchicalSoftmaxLoss

parser = argparse.ArgumentParser(description='COMP90051 Project1')
# Training
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--l2_reg', type=float, default=0.00025)
parser.add_argument('--num_of_group', type=int, default=9295)
parser.add_argument('--num_of_classes', type=int, default=95)
parser.add_argument('--epoch', type=int, default=310)
parser.add_argument('--grad_clip_bound', type=float, default=5.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--default_bert', action='store_true', default=False)
parser.add_argument('--features_extract', action='store_true', default=False)
parser.add_argument('--use_group', action='store_true', default=False)
parser.add_argument('--feature_size', type=int, default=192)
parser.add_argument('--center_loss_alpha', type=float, default=0.1)
parser.add_argument('--center_loss_lr', type=float, default=0.01)

# FilePath
parser.add_argument('--check_point_path', type=str, default="checkpoints/")
parser.add_argument('--model_version_string', type=str, default="coconut_model_v1")
parser.add_argument('--train_set_file_path', type=str, default="data/v4/train_set.txt")
parser.add_argument('--dev_set_file_path', type=str, default="data/v4/dev_set.txt")
parser.add_argument('--idx_file_path', type=str, default="data/v4/idx.pickle")
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
                               num_of_classes=args.num_of_classes,
                               use_group=args.use_group)

    dev_set = ProjectDataset(file_path=args.dev_set_file_path,
                             idx_file_path=args.idx_file_path,
                             num_of_classes=args.num_of_classes,
                             use_group=args.use_group)

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


def save_mode(epoch, model, optimizer, lr_scheduler, center_loss=None):
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
        "lr_scheduler": lr_scheduler.state_dict(),
        "center_loss_state_dict": center_loss_state_dict,
        "global_step": GLOBAL_STEP,
        "CURRENT_BEST": CURRENT_BEST
    }
    torch.save(payload, filename)
    print('%s Saved!' % (filename))

    if CURRENT_ACC > CURRENT_BEST:
        CURRENT_BEST = CURRENT_ACC
        filename = args.check_point_path + args.model_version_string + '_best.pth'
        torch.save(payload, filename)
        print('%s Saved!' % (filename))

    return


def load_model(model, optimizer, lr_scheduler, center_loss=None):
    global GLOBAL_STEP, CURRENT_BEST
    filename = args.check_point_path + args.model_version_string + '.pth'
    checkpoints = torch.load(filename)
    epoch = checkpoints["epoch"]
    model.load_state_dict(checkpoints["model_state_dict"])
    # optimizer.load_state_dict(checkpoints["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoints["lr_scheduler"])
    lr_scheduler.optimizer = optimizer
    GLOBAL_STEP = checkpoints["global_step"]
    CURRENT_BEST = checkpoints["CURRENT_BEST"]
    if center_loss is not None:
        center_loss.load_state_dict(checkpoints["center_loss_state_dict"])
    print('%s Loaded!' % (filename))
    return epoch, model, optimizer, lr_scheduler, center_loss


def prepare_data_for_coconut_model(batch, bert_model, tokenizer):
    if args.use_group:
        sentences, group_id, id_in_group = batch
        group_id = group_id.to(device)
        id_in_group = id_in_group.to(device)
    else:
        sentences, labels = batch
        labels = labels.to(device)
    sentences = list(sentences)

    tokens_tensor_batch = []
    for batch_sentences in sentences:
        tokenized_text = tokenizer.tokenize(batch_sentences.lower())
        if len(tokenized_text) > 512:
            # TODO: Drop Long Sequence for now
            tokenized_text = tokenized_text[:512]
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor(indexed_tokens).to(device)
        tokens_tensor_batch.append(tokens_tensor)

    tokens_tensor = torch.nn.utils.rnn.pad_sequence(tokens_tensor_batch, batch_first=True)

    with torch.no_grad():
        bert_model.to(device)
        outputs = bert_model(tokens_tensor)
        print(len(outputs))
    if args.use_group:
        return outputs, group_id, id_in_group
    else:
        return outputs, labels


def eval_model(epoch, model, loader, bert_model, tokenizer):
    global CURRENT_ACC
    model.eval()
    model.to(device)
    if not args.default_bert:
        model.batch(True)

    train_group_acc_meter = AverageMeter()
    train_id_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    print('=' * 20 + "Model Eval" + '=' * 20)
    for i, batch in tqdm(enumerate(loader)):
        start = time.time()

        batch = prepare_data_for_coconut_model(batch, bert_model, tokenizer)

        with torch.no_grad():
            if args.use_group:
                input, group_id_labels, id_in_group_labels = batch
                group_pred, id_in_group_pred = model(input)
                loss = HierarchicalSoftmaxLoss(group_pred, id_in_group_pred, group_id_labels, id_in_group_labels)

                # Calculate Correct Count
                id_in_group_pred = torch.argmax(id_in_group_pred, dim=2).tolist()
                group_pred = torch.argmax(group_pred, dim=1).tolist()

                group_id_labels = group_id_labels.tolist()
                id_in_group_labels = id_in_group_labels.tolist()

                correct_count = 0
                group_correct_count = 0
                for batch in range(len(id_in_group_labels)):
                    id_label = id_in_group_labels[batch]
                    group_id_label = group_id_labels[batch]
                    group_id_pred = group_pred[batch]
                    id_pred = id_in_group_pred[batch][group_id_pred]
                    if group_id_pred == group_id_label:
                        group_correct_count += 1
                        if id_pred == id_label:
                            correct_count += 1

                train_group_acc_meter.update(group_correct_count / len(id_in_group_labels))
                train_id_acc_meter.update(correct_count / len(id_in_group_labels))
                loss_meter.update(loss.item())

                end = time.time()
                used_time = end - start

                display = 'epoch=' + str(epoch) + \
                          '\tglobal_step=%d' % (GLOBAL_STEP) + \
                          '\tloss=%.5f' % (loss_meter.val) + \
                          '\tloss_avg=%.5f' % (loss_meter.avg) + \
                          '\tgroup_acc=%.4f' % (train_group_acc_meter.avg) + \
                          '\tid_acc=%.4f' % (train_id_acc_meter.avg) + \
                          '\ttime=%.2fit/s' % (1. / used_time)

            else:
                input, labels = batch
                pred = model(input)
                loss = nn.CrossEntropyLoss()(pred, labels)
                train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
                train_id_acc_meter.update(train_acc.item())
                loss_meter.update(loss.item())
                end = time.time()
                used_time = end - start
                display = 'epoch=' + str(epoch) + \
                          '\tglobal_step=%d' % (GLOBAL_STEP) + \
                          '\tloss=%.5f' % (loss_meter.val) + \
                          '\tloss_avg=%.5f' % (loss_meter.avg) + \
                          '\tacc=%.4f' % (train_id_acc_meter.avg) + \
                          '\ttime=%.2fit/s' % (1. / used_time)

        if (i) % (args.log_every / 2) == 0:
            tqdm.write(display)

    print("Final Eval Acc: %.4f\n" % (train_id_acc_meter.avg))
    CURRENT_ACC = train_id_acc_meter.avg
    return


def train_model(epoch, model, optimizer, loader, bert_model, tokenizer, center_loss=None):
    global GLOBAL_STEP

    train_group_acc_meter = AverageMeter()
    train_id_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    model.train()
    model.to(device)

    if not args.default_bert:
        model.batch(True)

    print('=' * 20 + "Model Training" + '=' * 20)

    for i, batch in tqdm(enumerate(loader)):
        start = time.time()

        optimizer.zero_grad()
        model.zero_grad()
        batch = prepare_data_for_coconut_model(batch, bert_model, tokenizer)
        if args.use_group:
            input, group_id_labels, id_in_group_labels = batch
            group_pred, id_in_group_pred = model(input)
            loss = HierarchicalSoftmaxLoss(group_pred, id_in_group_pred, group_id_labels, id_in_group_labels)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_bound)
            optimizer.step()

            # Calculate Correct Count
            id_in_group_pred = torch.argmax(id_in_group_pred, dim=2).tolist()
            group_pred = torch.argmax(group_pred, dim=1).tolist()

            group_id_labels = group_id_labels.tolist()
            id_in_group_labels = id_in_group_labels.tolist()

            correct_count = 0
            group_correct_count = 0
            for batch in range(len(id_in_group_labels)):
                id_label = id_in_group_labels[batch]
                group_id_label = group_id_labels[batch]
                group_id_pred = group_pred[batch]
                id_pred = id_in_group_pred[batch][group_id_pred]
                if group_id_pred == group_id_label:
                    group_correct_count += 1
                    if id_pred == id_label:
                        correct_count += 1

            train_group_acc_meter.update(group_correct_count / len(id_in_group_labels))
            train_id_acc_meter.update(correct_count / len(id_in_group_labels))
            loss_meter.update(loss.item())

            end = time.time()
            used_time = end - start

            lr = optimizer.param_groups[0]['lr']
            display = 'epoch=' + str(epoch) + \
                      '\tglobal_step=%d' % (GLOBAL_STEP) + \
                      '\tloss=%.5f' % (loss_meter.val) + \
                      '\tloss_avg=%.5f' % (loss_meter.avg) + \
                      '\tlr=%.6f' % (lr) + \
                      '\t|g|=%.4f' % (grad_norm) + \
                      '\tgroup_acc=%.4f' % (train_group_acc_meter.avg) + \
                      '\tid_acc=%.4f' % (train_id_acc_meter.avg) + \
                      '\ttime=%.2fit/s' % (1. / used_time)
        else:
            input, labels = batch
            pred = model(input)
            loss = nn.CrossEntropyLoss()(pred, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_bound)
            optimizer.step()
            train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
            train_id_acc_meter.update(train_acc.item())
            loss_meter.update(loss.item())
            end = time.time()
            used_time = end - start
            lr = optimizer.param_groups[0]['lr']
            display = 'epoch=' + str(epoch) + \
                      '\tglobal_step=%d' % (GLOBAL_STEP) + \
                      '\tloss=%.5f' % (loss_meter.val) + \
                      '\tloss_avg=%.5f' % (loss_meter.avg) + \
                      '\tlr=%.6f' % (lr) + \
                      '\t|g|=%.4f' % (grad_norm) + \
                      '\tacc=%.4f' % (train_id_acc_meter.avg) + \
                      '\ttime=%.2fit/s' % (1. / used_time)

        if (GLOBAL_STEP) % args.log_every == 0:
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
    coconut_model = CoconutModel(num_of_classes=args.num_of_classes,
                                     num_of_group=args.num_of_group,
                                     feature_size=args.feature_size)

    if torch.cuda.is_available():
        coconut_model = coconut_model.cuda()
        bert_model = bert_model.cuda()

    optimizer = RAdam(params=coconut_model,
                      lr=args.lr,
                      betas=(0.0, 0.999),
                      eps=1e-3,
                      weight_decay=args.l2_reg)

    # optimizer = torch.optim.SGD(params,
    #                             lr=args.lr,
    #                             momentum=0.9,
    #                             nesterov=True,
    #                             weight_decay=args.l2_reg)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=[80, 150],
                                                        gamma=0.1)

    starting_epoch = 0

    if args.resume:
        checkpoints = load_model(model=coconut_model,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 center_loss=center_loss)
        (starting_epoch, coconut_model, optimizer, lr_scheduler, center_loss) = checkpoints

    for epoch in range(starting_epoch, args.epoch):
        train_model(epoch=epoch,
                    model=coconut_model,
                    optimizer=optimizer,
                    loader=data_loaders["train_loader"],
                    tokenizer=tokenizer,
                    bert_model=bert_model,
                    center_loss=center_loss)
        lr_scheduler.step()
        eval_model(epoch=epoch,
                   model=coconut_model,
                   loader=data_loaders["dev_loader"],
                   tokenizer=tokenizer,
                   bert_model=bert_model)
        save_mode(epoch=epoch,
                  model=coconut_model,
                  optimizer=optimizer,
                  lr_scheduler=lr_scheduler,
                  center_loss=center_loss)
    return


if __name__ == "__main__":
    train()
