import torch
from torch import nn

from typing import Dict
from segmentation_models_pytorch.losses import DiceLoss
from src.utils.dataflow import ann_to_embedding, ann_to_one_hot


def nxn_cos_sim(A, B, dim=1, eps=1e-8):
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)


class HDLoss(nn.Module):
    def __init__(self, embeddigns: torch.Tensor, logit_scale: float):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # learnable temperature parameter
        self.logit_scale = torch.nn.Parameter(logit_scale)
        self.logit_scale.requires_grad = True
        # self.logit_scale = logit_scale
        self.embeddings = embeddigns

    def __call__(
        self, output: torch.Tensor, ann_one_hot: torch.Tensor
    ):
        bs, channels, h, w = output.shape
        output = output.permute(0, 2, 3, 1).view(-1, channels)
        cos_sim = nxn_cos_sim(output, self.embeddings)
        ann_one_hot = ann_one_hot.permute(0, 2, 3, 1).reshape(-1, ann_one_hot.shape[1])
        loss = self.log_softmax(cos_sim / self.logit_scale)
        loss = loss * ann_one_hot
        loss = -loss.mean()
        return loss

class CustomDiceLoss(nn.Module):
    def __init__(self, embeddigns: torch.Tensor):
        super().__init__()
        self.criterion = DiceLoss(mode='multilabel', smooth=0.0001, from_logits=False)
        self.embeddings = embeddigns

    def __call__(
        self, output: torch.Tensor, ann_one_hot: torch.Tensor
    ):

        bs, channels, h, w = output.shape
        output = output.permute(0, 2, 3, 1).view(-1, channels)
        cos_sim = nxn_cos_sim(output, self.embeddings)
        # get the class indices for max values along the 'num_cls' dim
        _, labels = cos_sim.max(dim=-1) 
        # get the most probable class for each pixel
        labels = labels.view(bs, h, w)
        pred_one_hot = ann_to_one_hot(labels, self.embeddings.shape[0])

        dice_loss = self.criterion(pred_one_hot, ann_one_hot)

        return dice_loss

