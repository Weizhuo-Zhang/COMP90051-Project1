import argparse
import time
import torch
from tqdm import tqdm
from models.coconut_model_v9 import CoconutModel
from project_dataset import ProjectDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from utils.utils import AverageMeter
from radam import RAdam
from HierarchicalSoftmax import HierarchicalSoftmaxLoss

parser = argparse.ArgumentParser(description='COMP90051 Project1')
# Training
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--l2_reg', type=float, default=2.5e-06)
parser.add_argument('--margin', type=float, default=1.0)
parser.add_argument('--epoch', type=int, default=310)
parser.add_argument('--grad_clip_bound', type=float, default=5.0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_of_group', type=int, default=98)
parser.add_argument('--num_of_classes', type=int, default=95)

# FilePath
parser.add_argument('--check_point_path', type=str, default="checkpoints/")
parser.add_argument('--model_version_string', type=str, default="coconut_model_v1")
parser.add_argument('--train_set_file_path', type=str, default="data/v3/train_set.txt")
parser.add_argument('--dev_set_file_path', type=str, default="data/v3/dev_set.txt")
parser.add_argument('--idx_file_path', type=str, default="data/v3/idx.pickle")
parser.add_argument('--log_every', type=int, default=100)

args = parser.parse_known_args()[0]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

GLOBAL_STEP = 0
CURRENT_ACC = 0
CURRENT_BEST_ACC = 20


for arg in vars(args):
    print(arg, getattr(args, arg))


def get_data_loader():
    data_loaders = {}

    train_set = ProjectDataset(file_path=args.train_set_file_path,
                               idx_file_path=args.idx_file_path,
                               use_group=True)

    dev_set = ProjectDataset(file_path=args.dev_set_file_path,
                             idx_file_path=args.idx_file_path,
                             use_group=True)

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
    optimizer.load_state_dict(checkpoints["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoints["lr_scheduler"])
    lr_scheduler.optimizer = optimizer
    GLOBAL_STEP = checkpoints["global_step"]
    CURRENT_BEST_ACC = checkpoints["CURRENT_BEST_ACC"]
    if center_loss is not None:
        center_loss.load_state_dict(checkpoints["center_loss_state_dict"])
    print('%s Loaded!' % (filename))
    return epoch, model, optimizer, lr_scheduler, center_loss


def eval_model(epoch, model, loader):
    global CURRENT_ACC
    model.eval()
    model.to(device)

    train_group_acc_meter = AverageMeter()
    train_id_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    loss_func = HierarchicalSoftmaxLoss

    print('=' * 20 + "Model Eval" + '=' * 20)
    for i, batch in tqdm(enumerate(loader)):
        start = time.time()
        with torch.no_grad():
            sentnece, group_label, id_label = batch
            group_label, id_label = group_label.to(device), id_label.to(device)
            group_pred, id_pred = model(sentnece)
            loss = loss_func(group_pred, id_pred, group_label, id_label)

            # Calculate Correct Count
            id_pred = torch.argmax(id_pred, dim=2).tolist()
            group_pred = torch.argmax(group_pred, dim=1).tolist()

            group_id_labels = group_label.tolist()
            id_in_group_labels = id_label.tolist()

            correct_count = 0
            group_correct_count = 0
            for batch in range(len(id_in_group_labels)):
                id_label = id_in_group_labels[batch]
                group_id_label = group_id_labels[batch]
                group_id_pred = group_pred[batch]
                id_pred = id_pred[batch][group_id_pred]
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

        if (i) % (args.log_every / 2) == 0:
            tqdm.write(display)
    print("Final Eval Acc: %.6f" % (train_id_acc_meter.avg))
    CURRENT_ACC = train_id_acc_meter.avg
    return


def train_model(epoch, model, optimizer, lr_scheduler, loader, center_loss=None):
    global GLOBAL_STEP

    train_group_acc_meter = AverageMeter()
    train_id_acc_meter = AverageMeter()
    loss_meter = AverageMeter()

    model.train()
    model.to(device)

    print('=' * 20 + "Model Training" + '=' * 20)

    loss_func = HierarchicalSoftmaxLoss

    for i, batch in tqdm(enumerate(loader)):
        start = time.time()

        optimizer.zero_grad()
        model.zero_grad()
        sentnece, group_label, id_label = batch
        group_label, id_label = group_label.to(device), id_label.to(device)
        group_pred, id_pred = model(sentnece)
        loss = loss_func(group_pred, id_pred, group_label, id_label)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_bound)
        optimizer.step()
        # Calculate Correct Count
        id_pred_list = torch.argmax(id_pred, dim=2).tolist()
        group_pred_list = torch.argmax(group_pred, dim=1).tolist()

        group_id_labels = group_label.tolist()
        id_in_group_labels = id_label.tolist()

        correct_count = 0
        group_correct_count = 0
        for batch in range(len(id_in_group_labels)):
            id_label = id_in_group_labels[batch]
            group_id_label = group_id_labels[batch]
            group_id_pred = group_pred_list[batch]
            id_pred = id_pred_list[batch][group_id_pred]
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

        if (GLOBAL_STEP) % args.log_every == 0:
            tqdm.write(display)
            save_mode(epoch=epoch,
                      model=model,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      center_loss=center_loss)
        GLOBAL_STEP += 1
    return


def train():
    # Init Training
    data_loaders = get_data_loader()
    coconut_model = CoconutModel()

    coconut_model.to(device)
    center_loss = None

    params = list(coconut_model.parameters())

    optimizer = RAdam(params=params,
                      lr=args.lr,
                      betas=(0.0, 0.999),
                      eps=1e-3,
                      weight_decay=args.l2_reg)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                        milestones=[1, 3, 5],
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
                    lr_scheduler=lr_scheduler,
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
                  center_loss=center_loss,
                  display=True)
    return


if __name__ == "__main__":
    train()
