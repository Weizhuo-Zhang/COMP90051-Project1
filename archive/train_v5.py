import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from models.coconut_model_v5 import CoconutModel
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
parser.add_argument('--num_of_classes', type=int, default=1132)
parser.add_argument('--epoch', type=int, default=310)
parser.add_argument('--grad_clip_bound', type=float, default=5.0)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--feature_size', type=int, default=192)
parser.add_argument('--center_loss_alpha', type=float, default=0.1)
parser.add_argument('--center_loss_lr', type=float, default=0.01)

# FilePath
parser.add_argument('--check_point_path', type=str, default="checkpoints/")
parser.add_argument('--model_version_string', type=str, default="coconut_model_v5")
parser.add_argument('--train_set_file_path', type=str, default="data/feature_extract_data_1132/train_set.txt")
parser.add_argument('--dev_set_file_path', type=str, default="data/feature_extract_data_1132/dev_set.txt")
parser.add_argument('--idx_file_path', type=str, default="data/feature_extract_data_1132/idx.pickle")
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
                               num_of_classes=args.num_of_classes,
                               idx_file_path=args.idx_file_path)

    dev_set = ProjectDataset(file_path=args.dev_set_file_path,
                             num_of_classes=args.num_of_classes,
                             idx_file_path=args.idx_file_path)

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
    optimizer.load_state_dict(checkpoints["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoints["lr_scheduler"])
    lr_scheduler.optimizer = optimizer
    GLOBAL_STEP = checkpoints["global_step"]
    CURRENT_BEST = checkpoints["CURRENT_BEST"]
    if center_loss is not None:
        center_loss.load_state_dict(checkpoints["center_loss_state_dict"])
    print('%s Loaded!' % (filename))
    return epoch, model, optimizer, lr_scheduler, center_loss


def eval_model(epoch, model, loader):
    global CURRENT_ACC
    model.eval()
    model.to(device)

    acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    print('=' * 20 + "Model Eval" + '=' * 20)
    for i, batch in tqdm(enumerate(loader)):
        start = time.time()
        with torch.no_grad():
            sentences, labels = batch
            labels = labels.to(device)
            _, pred = model(sentences)
            loss = nn.CrossEntropyLoss()(pred, labels)
            train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
            acc_meter.update(train_acc.item())
            loss_meter.update(loss.item())
            end = time.time()
            used_time = end - start
            display = 'epoch=' + str(epoch) + \
                      '\tglobal_step=%d' % (GLOBAL_STEP) + \
                      '\tloss=%.5f' % (loss_meter.val) + \
                      '\tloss_avg=%.5f' % (loss_meter.avg) + \
                      '\tacc=%.4f' % (acc_meter.avg) + \
                      '\ttime=%.2fit/s' % (1. / used_time)

        if (i) % (args.log_every / 2) == 0:
            tqdm.write(display)
    print("Final Eval Acc: %.4f\n" % (acc_meter.avg))
    CURRENT_ACC = acc_meter.avg
    return


def train_model(epoch, model, optimizer, loader, center_loss=None):
    global GLOBAL_STEP

    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    c_loss_meter = AverageMeter()
    model.train()
    model.to(device)

    print('=' * 20 + "Model Training" + '=' * 20)

    for i, batch in tqdm(enumerate(loader)):
        start = time.time()

        optimizer.zero_grad()
        model.zero_grad()
        sentences, labels = batch
        labels = labels.to(device)
        feature, pred = model(sentences)
        c_loss = center_loss(feature, labels) * args.center_loss_alpha
        ce_loss = nn.CrossEntropyLoss()(pred, labels)
        loss = c_loss + ce_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_bound)
        for param in center_loss.parameters():
            param.grad.data *= (args.center_loss_lr / (args.center_loss_alpha * args.lr))
        optimizer.step()
        train_acc = torch.mean((torch.max(pred, 1)[1] == labels).type(torch.float))
        acc_meter.update(train_acc.item())
        loss_meter.update(loss.item())
        ce_loss_meter.update(ce_loss.item())
        c_loss_meter.update(c_loss.item())
        end = time.time()
        used_time = end - start
        lr = optimizer.param_groups[0]['lr']
        display = 'epoch=' + str(epoch) + \
                  '\tglobal_step=%d' % (GLOBAL_STEP) + \
                  '\tloss=%.5f' % (loss_meter.val) + \
                  '\tloss_avg=%.5f' % (loss_meter.avg) + \
                  '\tce_loss=%.4f' % (ce_loss_meter.avg) + \
                  '\tc_loss=%.4f' % (c_loss_meter.avg) + \
                  '\tlr=%.6f' % (lr) + \
                  '\t|g|=%.4f' % (grad_norm) + \
                  '\tacc=%.4f' % (acc_meter.avg) + \
                  '\ttime=%.2fit/s' % (1. / used_time)
        if (GLOBAL_STEP) % args.log_every == 0:
            tqdm.write(display)
        GLOBAL_STEP += 1
    return


def train():
    # Init Training
    data_loaders = get_data_loader()
    coconut_model = CoconutModel(num_of_classes=1132,
                                 feature_size=args.feature_size)

    center_loss = CenterLoss(num_classes=1132,
                             feat_dim=192,
                             use_gpu=torch.cuda.is_available())

    coconut_model.to(device)
    center_loss.to(device)

    params = list(coconut_model.parameters()) + list(center_loss.parameters()) + list(coconut_model.bert_model.parameters())

    optimizer = RAdam(params=params,
                      lr=args.lr,
                      betas=(0.0, 0.999),
                      eps=1e-3,
                      weight_decay=args.l2_reg)

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
                    center_loss=center_loss)
        lr_scheduler.step()
        eval_model(epoch=epoch,
                   model=coconut_model,
                   loader=data_loaders["dev_loader"])
        save_mode(epoch=epoch,
                  model=coconut_model,
                  optimizer=optimizer,
                  lr_scheduler=lr_scheduler,
                  center_loss=center_loss)
    return


if __name__ == "__main__":
    train()
