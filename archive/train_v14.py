import argparse
import time
import torch
from tqdm import tqdm
from models.coconut_model_v14 import CoconutModel
from project_dataset_triplet import ProjectDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from utils.utils import AverageMeter
from pytorch_transformers import AdamW
from triplet_loss import TripletLoss

parser = argparse.ArgumentParser(description='COMP90051 Project1')
# Training
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--l2_reg', type=float, default=2.5e-05)
parser.add_argument('--margin', type=float, default=1.0)
parser.add_argument('--epoch', type=int, default=310)
parser.add_argument('--grad_clip_bound', type=float, default=5.0)
parser.add_argument('--batch_size', type=int, default=2)

# FilePath
parser.add_argument('--check_point_path', type=str, default="checkpoints/")
parser.add_argument('--model_version_string', type=str, default="coconut_model_v14")
parser.add_argument('--log_every', type=int, default=100)

args = parser.parse_known_args()[0]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

GLOBAL_STEP = 0
CURRENT_ACC = 0
CURRENT_BEST_ACC = 0


for arg in vars(args):
    print(arg, getattr(args, arg))


def get_data_loader():
    data_loaders = {}

    data_set = ProjectDataset(file_path="data/triplet/triple_sentences.csv")
    total = len(data_set)
    train_indices = list(range(0, int(total * 0.9)))
    valid_indices = list(range(int(total * 0.9), len(data_set)))
    train_set = Subset(data_set, train_indices)
    valid_set = Subset(data_set, valid_indices)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=4)

    dev_loader = DataLoader(valid_set,
                            batch_size=args.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=4)

    data_loaders["train_loader"] = train_loader
    data_loaders["dev_loader"] = dev_loader
    return data_loaders


def save_mode(epoch, model, optimizer, lr_scheduler, center_loss=None, display=False):
    global GLOBAL_STEP, CURRENT_ACC, CURRENT_BEST_ACC
    filename = args.check_point_path + args.model_version_string + '.pth'
    payload = {
        "epoch": epoch + 1,
        "args": args,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "global_step": GLOBAL_STEP,
        "CURRENT_BEST_ACC": CURRENT_BEST_ACC
    }
    torch.save(payload, filename)
    if display:
        print('%s Saved!' % (filename))

    if CURRENT_ACC > CURRENT_BEST_ACC:
        CURRENT_BEST_ACC = CURRENT_ACC
        filename = args.check_point_path + args.model_version_string + '_best.pth'
        torch.save(payload, filename)
        if display:
            print('%s Saved!' % (filename))
    return


def load_model(model, optimizer, lr_scheduler, center_loss=None):
    global GLOBAL_STEP, CURRENT_BEST_ACC
    filename = args.check_point_path + args.model_version_string + '.pth'
    checkpoints = torch.load(filename)
    epoch = checkpoints["epoch"]
    model.load_state_dict(checkpoints["model_state_dict"])
    # optimizer.load_state_dict(checkpoints["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoints["lr_scheduler"])
    lr_scheduler.optimizer = optimizer
    GLOBAL_STEP = checkpoints["global_step"]
    CURRENT_BEST_ACC = checkpoints["CURRENT_BEST_ACC"]
    if center_loss is not None:
        center_loss.load_state_dict(checkpoints["center_loss_state_dict"])
    print('%s Loaded!' % (filename))
    return epoch, model, optimizer, lr_scheduler, center_loss


def batch_eval(batch, model):
    model.eval()
    triplet_loss_func = TripletLoss(margin=args.margin)

    anchor, positive, negative = batch
    anchor_feature = model(anchor)
    positive_feature = model(positive)
    negative_feature = model(negative)
    triplet_loss = triplet_loss_func(anchor_feature, positive_feature, negative_feature)
    loss = triplet_loss

    positive_distance = (anchor_feature - positive_feature).pow(2).sum(1).detach()
    negative_distance = (anchor_feature - negative_feature).pow(2).sum(1).detach()
    acc = torch.gt(negative_distance, positive_distance).type(torch.FloatTensor).mean()
    return loss, acc, positive_distance, negative_distance


def eval_model(epoch, model, loader):
    global CURRENT_ACC
    model.eval()
    model.to(device)

    positive_distance_meter = AverageMeter()
    negative_distance_meter = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    print('=' * 20 + "Model Eval" + '=' * 20)
    for i, batch in tqdm(enumerate(loader)):
        start = time.time()
        with torch.no_grad():
            loss, acc, positive_distance, negative_distance = batch_eval(batch, model)
            loss_meter.update(loss.item())
            acc_meter.update(acc.item())
            positive_distance_meter.update(positive_distance.mean().item())
            negative_distance_meter.update(negative_distance.mean().item())

        end = time.time()
        used_time = end - start
        if (i) % (args.log_every / 2) == 0:
            display = 'epoch=' + str(epoch) + \
                      '\tglobal_step=%d' % (GLOBAL_STEP) + \
                      '\tloss=%.4f' % (loss_meter.val) + \
                      '\tloss_avg=%.4f' % (loss_meter.avg) + \
                      '\tpos_avg=%.4f' % (positive_distance_meter.avg) + \
                      '\tneg_avg=%.4f' % (negative_distance_meter.avg) + \
                      '\tacc=%.4f' % (acc_meter.avg) + \
                      '\ttime=%.2fit/s' % (1. / used_time)
            tqdm.write(display)
    print("Final Acc: %.6f" % (acc_meter.avg))
    print("Final Loss Acc: %.6f" % (loss_meter.avg))
    print("Final Positive Distance: %.6f" % (positive_distance_meter.avg))
    print("Final Negative Distance: %.6f" % (negative_distance_meter.avg))
    CURRENT_ACC = acc_meter.avg
    return


def train_model(epoch, model, optimizer, lr_scheduler, loader, test_loader):
    global GLOBAL_STEP

    test_loader_it = iter(test_loader)

    loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    val_acc_meter = AverageMeter()
    positive_distance_meter = AverageMeter()
    negative_distance_meter = AverageMeter()

    model.train()
    model.to(device)

    print('=' * 20 + "Model Training" + '=' * 20)

    triplet_loss_func = TripletLoss(args.margin)

    for i, batch in tqdm(enumerate(loader)):
        start = time.time()
        model.train()
        optimizer.zero_grad()
        model.zero_grad()
        anchor, positive, negative = batch
        anchor_feature = model(anchor)
        positive_feature = model(positive)
        negative_feature = model(negative)
        triplet_loss = triplet_loss_func(anchor_feature, positive_feature, negative_feature)
        loss = triplet_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_bound)
        optimizer.step()

        positive_distance = (anchor_feature - positive_feature).pow(2).sum(1).detach()
        negative_distance = (anchor_feature - negative_feature).pow(2).sum(1).detach()
        acc = torch.gt(negative_distance, positive_distance).type(torch.FloatTensor).mean()
        loss_meter.update(loss.item())
        acc_meter.update(acc.item())
        positive_distance_meter.update(positive_distance.mean().item())
        negative_distance_meter.update(negative_distance.mean().item())

        end = time.time()
        used_time = end - start

        if (GLOBAL_STEP) % args.log_every == 0:
            try:
                batch = next(test_loader_it)
            except:
                test_loader_it = iter(test_loader)
                batch = next(test_loader_it)
            eval_loss, eval_acc, _, _ = batch_eval(batch, model)
            val_loss_meter.update(eval_loss.item())
            val_acc_meter.update(eval_acc.item())
            lr = optimizer.param_groups[0]['lr']
            display = 'epoch=' + str(epoch) + \
                      '\tglobal_step=%d' % (GLOBAL_STEP) + \
                      '\tloss=%.4f' % (loss_meter.val) + \
                      '\tloss_avg=%.4f' % (loss_meter.avg) + \
                      '\tval_loss=%.4f' % (val_loss_meter.avg) + \
                      '\tpos_avg=%.4f' % (positive_distance_meter.avg) + \
                      '\tneg_avg=%.4f' % (negative_distance_meter.avg) + \
                      '\tacc=%.4f' % (acc_meter.avg) + \
                      '\tval_acc=%.4f' % (val_acc_meter.avg) + \
                      '\tlr=%.6f' % (lr) + \
                      '\t|g|=%.4f' % (grad_norm) + \
                      '\ttime=%.2fit/s' % (1. / used_time)
            tqdm.write(display)
            save_mode(epoch=epoch,
                      model=model,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler)
        GLOBAL_STEP += 1
    return


def train():
    # Init Training
    data_loaders = get_data_loader()
    coconut_model = CoconutModel()

    coconut_model.to(device)
    center_loss = None

    params = list(coconut_model.parameters()) + list(coconut_model.bert_model.parameters())

    optimizer = AdamW(params=params,
                      lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=[3],
                                                        gamma=0.1)
    starting_epoch = 0

    if args.resume:
        checkpoints = load_model(model=coconut_model,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 center_loss=center_loss)
        (starting_epoch, coconut_model, optimizer, lr_scheduler, center_loss) = checkpoints

    # warmup_steps = int(len(data_loaders["train_loader"]))
    # print("WarmUp steps %d" % (warmup_steps))
    for epoch in range(starting_epoch, args.epoch):
        train_model(epoch=epoch,
                    model=coconut_model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    loader=data_loaders["train_loader"],
                    test_loader=data_loaders["dev_loader"])
        lr_scheduler.step()
        eval_model(epoch=epoch,
                   model=coconut_model,
                   loader=data_loaders["dev_loader"])
        save_mode(epoch=epoch,
                  model=coconut_model,
                  optimizer=optimizer,
                  lr_scheduler=lr_scheduler,
                  center_loss=center_loss,
                  display=True)
    return


if __name__ == "__main__":
    train()
