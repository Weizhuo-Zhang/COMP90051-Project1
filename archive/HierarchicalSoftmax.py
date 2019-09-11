import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Two Layer
def HierarchicalSoftmaxLoss(group_pred, id_pred, group_label, id_label):
    first_layer_softmax = F.softmax(group_pred, dim=1)
    second_layer_soft_max = F.softmax(id_pred, dim=2)

    # Build One Hot Vector For Group Label
    batch_size = group_label.size(0)
    group_size = group_pred.size(1)
    class_size = id_pred.size(2)
    group_label_onehot = torch.FloatTensor(batch_size, group_size).zero_().to(device)
    group_label_onehot = group_label_onehot.scatter_(1, group_label.view(-1, 1), 1)

    # First Layer Multiply
    first_layer_softmax = first_layer_softmax * group_label_onehot
    first_layer_softmax = first_layer_softmax.unsqueeze(2).expand(batch_size, group_size, class_size)

    # Build One Hot Vector For ID Labels
    batch_size = id_label.size(0)
    id_one_hot = torch.FloatTensor(batch_size, class_size).zero_().to(device)
    id_one_hot = id_one_hot.scatter_(1, id_label.view(-1, 1), 1)
    id_one_hot = id_one_hot.unsqueeze(1).expand(-1, group_size, class_size)
    second_layer_soft_max = second_layer_soft_max * id_one_hot

    loss = first_layer_softmax * second_layer_soft_max
    loss = -torch.log(loss.mean())
    return loss
