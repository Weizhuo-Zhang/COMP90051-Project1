import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


if __name__ == "__main__":
    target = torch.randn(1, 2)
    positive = torch.randn(1, 2)
    negative = torch.randn(1, 2)
    loss_func = TripletLoss(margin=0.2)
    loss = loss_func(target, positive, negative)
    print(target)
    print(positive)
    print((target - positive).pow(2).sum())
