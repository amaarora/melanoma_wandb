import torch.nn as nn
from torch.nn import functional as F
import torch
from efficientnet_pytorch import EfficientNet
from loss import WeightedFocalLoss


class EfficientNetBx(nn.Module):
    def __init__(self, pretrained=True, arch_name='efficientnet-b0', ce=False):
        super(EfficientNetBx, self).__init__()
        self.pretrained = pretrained
        self.ce = ce
        self.base_model = EfficientNet.from_pretrained(arch_name) if pretrained else EfficientNet.from_name(arch_name)
        nftrs = self.base_model._fc.in_features
        self.base_model._fc = nn.Sequential(
            *[nn.Linear(nftrs, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)])

    def forward(self, image, target, weights=None, args=None):
        out = self.base_model(image)  
        if args.loss=='weighted_bce' and weights is not None:
            weights_ = weights[target.data.view(-1).long()].view_as(target)
            loss_func = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_func(out, target.view(-1,1).type_as(out))
            loss_class_weighted = loss * weights_
            loss = loss_class_weighted.mean()
        elif args.loss == 'bce':
            loss = nn.BCEWithLogitsLoss()(out, target.view(-1,1).type_as(out))
        elif args.loss == 'weighted_focal_loss':
            loss = WeightedFocalLoss()(out, target.view(-1,1).type_as(out))
        else:
            raise ValueError("loss function not found.")
        return out, loss
