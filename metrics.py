import torch


def dice_score(pred, target, smooth=1e-5):
    """
    Calculates the Dice score as a metric for evaluation.
    This is the function to use for evaluating a trained model.

    Args:
        pred (torch.Tensor): The model's raw logits. Shape should be like (N, 1, H, W).
        target (torch.Tensor): The ground truth binary mask. Shape should match pred.
        smooth (float): A small value added to the denominator for numerical stability.

    Returns:
        torch.Tensor: The Dice score, a scalar tensor.
    """
    # Step 1: Apply sigmoid and a hard threshold to get a binary mask
    pred_probs = torch.sigmoid(pred)
    pred_mask = (pred_probs > 0.5).float()
    
    # Step 2: Flatten the tensors to 1D
    pred_flat = pred_mask.view(-1)
    target_flat = target.view(-1)
    
    # Step 3: Calculate the intersection
    intersection = (pred_flat * target_flat).sum()
    
    # Step 4: Calculate the Dice score
    score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return score