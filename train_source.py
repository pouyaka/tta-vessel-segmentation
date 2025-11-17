import torch
import torch.nn
import torch.nn.functional as F
torch.manual_seed(7)
import numpy as np
import albumentations as A
import sklearn.model_selection
import matplotlib.pyplot as plt
import csv
from torchmetrics.segmentation import DiceScore, MeanIoU
import os

import networks, dataset, losses


def main():
    checkpoint = False
    checkpoint_path = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LR = 5e-4
    input_h, input_w = 512, 512
    SOURCE_DATASET = 'CHASE'
    DATA_DIR = './CHASE/'
    IMG_EXT = 'jpg', 'JPG', 'png'
    MASK_EXT = 'tif', 'png'
    BATCH_SIZE = 8 
    NUM_WORKERS = 2
    EPOCHS = 100

    if checkpoint:
        name = 'fine_tuned'
    else:
        name = 'source'
    file_name = f"unet_{SOURCE_DATASET}_{name}"
    log_filename = 'training_log_' + file_name + '.csv'
    model_filename = file_name + '_weights.pth'

    model = networks.UNet(in_channels=3, out_channels=1)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    model = model.to(device)
    optim = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr= LR)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optim,
                                                    base_lr= 5e-5,
                                                    max_lr= LR,
                                                    step_size_up=8,)
    loss_fn = losses.BCEDiceLoss()
    train_dice = DiceScore(num_classes=2, include_background= False, average= 'macro').to(device)
    val_dice = DiceScore(num_classes=2, include_background= False, average= 'macro').to(device)
    val_iou = MeanIoU(num_classes=2, include_background= False).to(device)
    
    train_transform = A.Compose([
        A.Resize(input_h, input_w),
        A.RandomBrightnessContrast(p=0.3),
        A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.3),
        #A.GaussNoise(std_range=(0.1, 0.2), p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = A.Compose([
        A.Resize(input_h, input_w),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = dataset.FundusDataset(dataset_name= SOURCE_DATASET,
                                         data_dir= DATA_DIR+'train/' if SOURCE_DATASET == 'RITE' else DATA_DIR,
                                         img_ext= IMG_EXT, mask_ext= MASK_EXT,
                                         transform= train_transform)
    val_dataset = dataset.FundusDataset(dataset_name= SOURCE_DATASET,
                                         data_dir= DATA_DIR+'validation/' if SOURCE_DATASET == 'RITE' else DATA_DIR,
                                         img_ext= IMG_EXT, mask_ext= MASK_EXT,
                                         transform= val_transform)
    if SOURCE_DATASET == 'RITE':
        test_rite = dataset.FundusDataset(dataset_name= "RITE", data_dir= './RITE/test/',
                                        img_ext= IMG_EXT, mask_ext= MASK_EXT,
                                        transform= val_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle= True,
                                                 num_workers= NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= BATCH_SIZE)
        test_loader = torch.utils.data.DataLoader(test_rite, batch_size= BATCH_SIZE)
    else:
        num_data = len(train_dataset)
        indices = list(range(num_data))
        train_indices, val_indices = sklearn.model_selection.train_test_split(
            indices,
            test_size= 0.2,
            random_state= 7
        )
        train_data = torch.utils.data.Subset(train_dataset, train_indices)
        val_data = torch.utils.data.Subset(val_dataset, val_indices)

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size= BATCH_SIZE, 
                                                   shuffle= True, 
                                                   num_workers= NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size= BATCH_SIZE)
    

    for epoch in range(EPOCHS):
        # train step
        model.train()
        train_loss = 0.0

        for X,Y in train_loader:
            optim.zero_grad()
            X = X.to(device)
            Y = Y.to(device)
            pred = model(X)
            loss = loss_fn(pred, Y)
            
            loss.backward()
            optim.step()
            scheduler.step()

            if loss is not None:
                train_loss += loss.item()
            
            pred_probs = torch.sigmoid(pred)
            pred_mask = (pred_probs > 0.5).float()
            pred_onehot = F.one_hot(pred_mask.long().squeeze(1), num_classes=2).permute(0, 3, 1, 2).float()
            target_onehot = F.one_hot(Y.long().squeeze(1), num_classes=2).permute(0, 3, 1, 2).float()
            train_dice.update(pred_onehot, target_onehot)
        
        epoch_train_dice = train_dice.compute().item()
        train_dice.reset()
        
        # val step
        with torch.no_grad():
            model.eval()
            val_loss = 0.0

            for val_batch, (X,Y) in enumerate(val_loader):
                X = X.cuda()
                Y = Y.cuda()
                pred = model(X)
                loss = loss_fn(pred, Y)
                
                if loss is not None:
                    val_loss += loss.item()
                
                pred_probs = torch.sigmoid(pred)
                pred_mask = (pred_probs > 0.5).float()
                pred_onehot = F.one_hot(pred_mask.long().squeeze(1), num_classes=2).permute(0, 3, 1, 2).float()
                target_onehot = F.one_hot(Y.long().squeeze(1), num_classes=2).permute(0, 3, 1, 2).float()
                val_dice.update(pred_onehot, target_onehot)
                val_iou.update(pred_onehot, target_onehot)
                # Track entropy
                #probs = torch.sigmoid(pred)
                #epsilon = 1e-7
                #entropy = -(probs * torch.log(probs + epsilon) + 
                #            (1 - probs) * torch.log(1 - probs + epsilon))

        epoch_val_dice = val_dice.compute()
        epoch_val_iou = val_iou.compute().item()
        val_dice.reset()
        val_iou.reset()
        current_lr = optim.param_groups[0]['lr']
        epoch_info = {
            'epoch': epoch+1,
            'lr': round(current_lr,6),
            'train_loss': round(train_loss/len(train_loader), 4),
            'train dice': round(epoch_train_dice, 4),
            'val_loss': round(val_loss/len(val_loader), 4),
            'val_dice': round(epoch_val_dice.item(), 4),
            'val_mIOU' : round(epoch_val_iou, 4),
        }
        print(epoch_info)

        file_exists = os.path.isfile(log_filename)
        with open(log_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=epoch_info.keys())

            # write header only once
            if not file_exists:
                writer.writeheader()

            writer.writerow(epoch_info)
        
    torch.save(model.state_dict(), model_filename)


if __name__ == '__main__':
    main()