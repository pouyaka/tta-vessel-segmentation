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

import networks, dataset, tent

# Test different configurations
LR = 1e-3
SOURCE_CHECKPOINT = 'unet_HRF_source_weights.pth'
input_h, input_w = 512, 512
IMG_EXT = 'jpg', 'JPG', 'png'
MASK_EXT = 'tif', 'png'
log_filename = 'tent_' + os.path.splitext(SOURCE_CHECKPOINT)[0] + '_log.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configs = [
    {
        'name': 'Source (No Adaptation) BS=1',
        'use_tent': False,
        'batch_size': 1,
        'lr': None,
        'tent_steps': None,
        'episodic': None,
    },
    {
        'name': 'Tent: Steps=1, episodic=False, BS=1 (Original)',
        'use_tent': True,
        'batch_size': 1,
        'lr': LR,
        'tent_steps': 1,
        'episodic': False,
    },
    {
        'name': 'Tent: Steps=3, episodic=False, BS=1',
        'use_tent': True,
        'batch_size': 1,
        'lr': LR,
        'tent_steps': 3,
        'episodic': False,
    },
    {
        'name': 'Tent: Steps=5, episodic=False, BS=1',
        'use_tent': True,
        'batch_size': 1,
        'lr': LR,
        'tent_steps': 5,
        'episodic': False,
    },
    {
        'name': 'Tent: Steps=1, episodic=True, BS=1',
        'use_tent': True,
        'batch_size': 1,
        'lr': LR,
        'tent_steps': 1,
        'episodic': True,
    },
    {
        'name': 'Tent: Steps=3, episodic=True, BS=1',
        'use_tent': True,
        'batch_size': 1,
        'lr': LR,
        'tent_steps': 3,
        'episodic': True,
    },
]

def main():

    net_checkpoint_path = SOURCE_CHECKPOINT
    net_cpu = networks.UNet(in_channels=3, out_channels=1)
    net_cpu.load_state_dict(torch.load(net_checkpoint_path, weights_only=True))
    model = net_cpu.cuda()

    eval_transform = A.Compose([
        A.Resize(input_h, input_w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    chase = dataset.FundusDataset(dataset_name= "CHASE", data_dir= './CHASE/',
                                        img_ext= IMG_EXT, mask_ext= MASK_EXT,
                                        transform= eval_transform)
    hrf = dataset.FundusDataset(dataset_name= "HRF", data_dir= './HRF/all/',
                                        img_ext= IMG_EXT, mask_ext= MASK_EXT,
                                        transform= eval_transform)
    train_rite = dataset.FundusDataset(dataset_name= "RITE", data_dir= './RITE/train/',
                                        img_ext= IMG_EXT, mask_ext= MASK_EXT,
                                        transform= eval_transform)
    val_rite = dataset.FundusDataset(dataset_name= "RITE", data_dir= './RITE/validation/',
                                        img_ext= IMG_EXT, mask_ext= MASK_EXT,
                                        transform= eval_transform)
    test_rite = dataset.FundusDataset(dataset_name= "RITE", data_dir= './RITE/test/',
                                        img_ext= IMG_EXT, mask_ext= MASK_EXT,
                                        transform= eval_transform)
    if 'HRF' in SOURCE_CHECKPOINT:
        eval_dataset = ConcatDataset([chase, train_rite, val_rite, test_rite])
    elif 'CHASE' in SOURCE_CHECKPOINT:
        eval_dataset = ConcatDataset([hrf, train_rite, val_rite, test_rite])
    elif 'RITE' in SOURCE_CHECKPOINT:
        eval_dataset = ConcatDataset([chase, hrf])
        
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1)

    parts = SOURCE_CHECKPOINT.split("_")
    print(f'\n Model: {parts[0]}, Dataset: {parts[1]}')

    for config in configs:

        if 'Source' in config['name']:
            source_model = deepcopy(model)
            source_model.eval()
            with torch.no_grad():
                results = eval_loop(source_model, eval_loader)
            dice_scores, iou_values, entropy_values = results
                
        else:
            t_model = tent.configure_model(model)
            params, param_names = tent.collect_params(t_model)
            optimizer = torch.optim.Adam(params, lr=config['lr'])
            tented_model = tent.Tent(t_model, optimizer, 
                                    steps= config['tent_steps'],
                                    episodic= config['episodic'])
            tent.check_model(tented_model)

            results = eval_loop(tented_model, eval_loader)
            dice_scores, iou_values, entropy_values = results
            tented_model.reset()

        info = {
            **config,
            'mean_dice': np.mean(dice_scores),
            'std_dice': np.std(dice_scores),
            'mean_iou': np.mean(iou_values),
            'std_iou': np.std(iou_values),
            'mean_entropy': np.mean(entropy_values),
            'std_entropy': np.std(entropy_values),
        }

        
        print(f'\n {config['name']}')
        print(f'  Learning rate:   {config['lr']}')
        print(f"  Dice:   {info['mean_dice']:.4f} ± {info['std_dice']:.4f}")
        print(f"  IOU:   {info['mean_iou']:.4f} ± {info['std_iou']:.4f}")
        print(f"  Entropy: {info['mean_entropy']:.4f}")

        file_exists = os.path.isfile(log_filename)
        with open(log_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=info.keys())

            # write header only once
            if not file_exists:
                writer.writeheader()

            writer.writerow(info)
            

def eval_loop(net, loader):

    entropy_values = []
    dice_scores = []
    iou_values = []

    dice = DiceScore(num_classes=2, include_background= False, average= 'macro').to(device)
    iou = MeanIoU(num_classes=2, include_background= False).to(device)

    for X,Y in loader:
        X = X.cuda()
        Y = Y.cuda()
        pred = net(X)
        pred_probs = torch.sigmoid(pred)
        pred_mask = (pred_probs > 0.5).float()
        pred_onehot = F.one_hot(pred_mask.long().squeeze(1), num_classes=2).permute(0, 3, 1, 2).float()
        target_onehot = F.one_hot(Y.long().squeeze(1), num_classes=2).permute(0, 3, 1, 2).float()
        dice_score = dice(pred_onehot, target_onehot)
        dice.reset()
        iou_value = iou(pred_onehot, target_onehot)
        iou.reset()
        epsilon = 1e-7
        entropy = -(pred_probs * torch.log(pred_probs + epsilon) + 
                    (1 - pred_probs) * torch.log(1 - pred_probs + epsilon))

        entropy_values.append(entropy.mean().item())
        dice_scores.append(dice_score.item())
        iou_values.append(iou_value.item())

    return dice_scores, iou_values, entropy_values


if __name__ == '__main__':
    main()