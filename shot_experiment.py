import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torchmetrics.segmentation import DiceScore, MeanIoU
import numpy as np
import albumentations as A
from tqdm import tqdm
from copy import deepcopy
import os
import csv

import networks, dataset, shot

# Test different configurations
LR = 1e-3
SOURCE_CHECKPOINT = 'checkpoints/unet_RITE_source_weights.pth'
input_h, input_w = 512, 512
IMG_EXT = 'jpg', 'JPG', 'png'
MASK_EXT = 'tif', 'png'
log_filename = 'shot' + os.path.splitext(SOURCE_CHECKPOINT)[0] + '_log.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configs = [
    {
        'name': 'Source (No Adaptation) BS=1',
        'use_shot': False,
        'batch_size': 8,
        'lr': None,
        'tent_steps': None,
        'episodic': None,
    },
    
    # SHOT configurations
    {
        'name': 'SHOT: Steps=1, episodic=False',
        'method': 'shot',
        'batch_size': 8,
        'lr': LR,
        'steps': 1,
        'episodic': False,
        'ent_weight': 0.5,
        'div_weight': 0.5,
    },
    {
        'name': 'SHOT: Steps=3, episodic=False',
        'method': 'shot',
        'batch_size': 8,
        'lr': LR,
        'steps': 3,
        'episodic': False,
        'ent_weight': 0.5,
        'div_weight': 0.5,
    },
    {
        'name': 'SHOT: Steps=5, episodic=False',
        'method': 'shot',
        'batch_size': 8,
        'lr': LR,
        'steps': 5,
        'episodic': False,
        'ent_weight': 0.5,
        'div_weight': 0.5,
    },
    {
        'name': 'SHOT: Steps=1, episodic=True',
        'method': 'shot',
        'batch_size': 8,
        'lr': LR,
        'steps': 1,
        'episodic': True,
        'ent_weight': 0.5,
        'div_weight': 0.5,
    },
    {
        'name': 'SHOT: Steps=3, episodic=True',
        'method': 'shot',
        'batch_size': 8,
        'lr': LR,
        'steps': 3,
        'episodic': True,
        'ent_weight': 0.5,
        'div_weight': 0.5,
    },
    {
        'name': 'SHOT: Steps=5, episodic=True',
        'method': 'shot',
        'batch_size': 8,
        'lr': LR,
        'steps': 5,
        'episodic': True,
        'ent_weight': 0.5,
        'div_weight': 0.5,
    },
]


def main():
    # Load source model
    net_checkpoint_path = SOURCE_CHECKPOINT
    net_cpu = networks.UNet(in_channels=3, out_channels=1)
    net_cpu.load_state_dict(torch.load(net_checkpoint_path, weights_only=True))
    model = net_cpu.cuda()

    # Prepare datasets
    eval_transform = A.Compose([
        A.Resize(input_h, input_w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    chase = dataset.FundusDataset(dataset_name="CHASE", data_dir='data/CHASE/',
                                  img_ext=IMG_EXT, mask_ext=MASK_EXT,
                                  transform=eval_transform)
    hrf = dataset.FundusDataset(dataset_name="HRF", data_dir='data/HRF/',
                                img_ext=IMG_EXT, mask_ext=MASK_EXT,
                                transform=eval_transform)
    train_rite = dataset.FundusDataset(dataset_name="RITE", data_dir='data/RITE/train/',
                                       img_ext=IMG_EXT, mask_ext=MASK_EXT,
                                       transform=eval_transform)
    val_rite = dataset.FundusDataset(dataset_name="RITE", data_dir='data/RITE/validation/',
                                     img_ext=IMG_EXT, mask_ext=MASK_EXT,
                                     transform=eval_transform)
    test_rite = dataset.FundusDataset(dataset_name="RITE", data_dir='data/RITE/test/',
                                      img_ext=IMG_EXT, mask_ext=MASK_EXT,
                                      transform=eval_transform)
    
    # Select evaluation dataset based on source
    if 'HRF' in SOURCE_CHECKPOINT:
        eval_dataset = ConcatDataset([chase, train_rite, val_rite, test_rite])
    elif 'CHASE' in SOURCE_CHECKPOINT:
        eval_dataset = ConcatDataset([hrf, train_rite, val_rite, test_rite])
    elif 'RITE' in SOURCE_CHECKPOINT:
        eval_dataset = ConcatDataset([chase, hrf])
    
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8)

    parts = SOURCE_CHECKPOINT.split("_")
    print(f'\nModel: {parts[0]}, Source Dataset: {parts[1]}')
    print(f'Evaluating on target domain with {len(eval_dataset)} images\n')
    print('=' * 80)

    # Run experiments for each configuration
    for config in configs:
        print(f'\nRunning: {config["name"]}')
        
        if 'Source' in config['name']:
            # Baseline: No adaptation
            source_model = deepcopy(model)
            source_model.eval()
            with torch.no_grad():
                results = eval_loop(source_model, eval_loader)
            dice_scores, iou_values, entropy_values = results
        
        elif config['method'] == 'shot':
            # SHOT method
            s_model = shot.configure_model(deepcopy(model))
            params, param_names = shot.collect_params(s_model)
            optimizer = torch.optim.Adam(params, lr=config['lr'])
            adapted_model = shot.SHOT(s_model, optimizer,
                                     steps=config['steps'],
                                     episodic=config['episodic'],
                                     ent_weight=config.get('ent_weight', 1.0),
                                     div_weight=config.get('div_weight', 1.0))
            shot.check_model(adapted_model)
            results = eval_loop(adapted_model, eval_loader)
            dice_scores, iou_values, entropy_values = results
            adapted_model.reset()
        
        # Compute statistics
        info = {
            **config,
            'mean_dice': np.mean(dice_scores),
            'std_dice': np.std(dice_scores),
            'mean_iou': np.mean(iou_values),
            'std_iou': np.std(iou_values),
            'mean_entropy': np.mean(entropy_values),
            'std_entropy': np.std(entropy_values),
        }
        
        # Print results
        print(f'  Dice:    {info["mean_dice"]:.4f} ± {info["std_dice"]:.4f}')
        print(f'  IoU:     {info["mean_iou"]:.4f} ± {info["std_iou"]:.4f}')
        print(f'  Entropy: {info["mean_entropy"]:.4f} ± {info["std_entropy"]:.4f}')
        
        # Save to CSV
        file_exists = os.path.isfile(log_filename)
        with open(log_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=info.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(info)
    
    print('\n' + '=' * 80)
    print(f'Results saved to: {log_filename}')


def eval_loop(net, loader):
    """Evaluation loop for any adapted model."""
    entropy_values = []
    dice_scores = []
    iou_values = []

    dice = DiceScore(num_classes=2, include_background=False, average='macro').to(device)
    iou = MeanIoU(num_classes=2, include_background=False).to(device)

    for X, Y in tqdm(loader, desc='Evaluating', leave=False):
        X = X.cuda()
        Y = Y.cuda()
        
        # Forward pass (adaptation happens inside if needed)
        pred = net(X)
        
        # Compute predictions
        pred_probs = torch.sigmoid(pred)
        pred_mask = (pred_probs > 0.5).float()
        
        # Convert to one-hot for metrics
        pred_onehot = F.one_hot(pred_mask.long().squeeze(1), num_classes=2).permute(0, 3, 1, 2).float()
        target_onehot = F.one_hot(Y.long().squeeze(1), num_classes=2).permute(0, 3, 1, 2).float()
        
        # Compute metrics
        dice_score = dice(pred_onehot, target_onehot)
        dice.reset()
        iou_value = iou(pred_onehot, target_onehot)
        iou.reset()
        
        # Compute entropy
        epsilon = 1e-7
        entropy = -(pred_probs * torch.log(pred_probs + epsilon) + 
                   (1 - pred_probs) * torch.log(1 - pred_probs + epsilon))
        
        entropy_values.append(entropy.mean().item())
        dice_scores.append(dice_score.item())
        iou_values.append(iou_value.item())

    return dice_scores, iou_values, entropy_values


if __name__ == '__main__':
    main()