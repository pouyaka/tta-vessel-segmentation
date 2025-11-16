import os
import torch.utils.data
import torchvision.transforms
import numpy as np
import cv2

class FundusDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, data_dir, img_ext, mask_ext,
                 transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── ...
            |
            └── masks
                ├── 0a7e06.png
                ├── 0aab0a.png
                ├── 0b1761.png
                ├── ...
        """
        input_dir = os.path.join(data_dir, 'images')
        self.img_ids = sorted(
            [
                os.path.join(input_dir, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(img_ext) and not fname.startswith(".")
            ]
        )
        

        if dataset_name == "HRF":
            target_data = os.path.join(data_dir, 'manual1')
            self.mask_ids = sorted(
                [
                    os.path.join(target_data, fname)
                    for fname in os.listdir(target_data)
                    if fname.endswith(mask_ext) and not fname.startswith(".")
                ]
            )
        elif dataset_name == "CHASE":
            target_data = os.path.join(data_dir, 'masks')
            self.mask_ids = sorted(
                [
                    os.path.join(target_data, fname)
                    for fname in os.listdir(target_data)
                    if fname.endswith('_2ndHO.png') and not fname.startswith(".") # _2ndHO.png, _1stHO.png
                ]
            )
        elif dataset_name == "RITE":
            target_data = os.path.join(data_dir, 'masks')
            self.mask_ids = sorted(
                [
                    os.path.join(target_data, fname)
                    for fname in os.listdir(target_data)
                    if fname.endswith(mask_ext) and not fname.startswith(".")
                ]
            )
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = cv2.imread(img_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_id = self.mask_ids[idx]
        mask = cv2.imread(mask_id, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        #img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)

        mask = mask.astype('float32') / 255
        mask = mask[np.newaxis, ...]

        
        return img, mask