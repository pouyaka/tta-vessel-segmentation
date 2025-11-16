import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss for segmentation tasks.
    This is particularly useful for imbalanced datasets.
    """
    
    def __init__(self, bce_weight=0.1, dice_weight=0.9, smooth=1e-7):
        """
        Args:
            bce_weight (float): Weight for BCE loss component
            dice_weight (float): Weight for Dice loss component
            smooth (float): Smoothing factor to avoid division by zero
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        
    def dice_loss(self, pred, target):
        """
        Calculate Dice loss
        Args:
            pred: Predictions from model (before sigmoid)
            target: Ground truth labels
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)
        
        # Flatten the tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        pred_sum = pred_flat.sum()
        target_sum = target_flat.sum()
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions from model (logits)
            target: Ground truth labels
        """
        # Ensure target is float and has same shape as pred
        target = target.float()
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        
        # Calculate Dice loss
        dice_loss_val = self.dice_loss(pred, target)
        
        # Combine both losses
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss_val
        
        return total_loss