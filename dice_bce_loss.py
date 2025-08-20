import torch
import torch.nn as nn

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0, alpha=0.5):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha  # weight between BCE and Dice

    def forward(self, inputs, targets):
        # BCE part
        bce_loss = self.bce(inputs, targets)

        # Dice part
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        # Hybrid
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss
