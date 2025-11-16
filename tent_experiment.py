import torch
import numpy as np
import albumentations as A
from tqdm import tqdm

import networks, dataset, metrics, tent

# Test different configurations
LR = 1e-5
configs = [
    {
        'name': 'Source (No Adaptation)',
        'use_tent': False,
        'batch_size': 1,
        'tent_lr': None,
        'tent_steps': None,
        'episodic': None,
    },
    {
        'name': 'Tent: Steps=1, episodic=False (Original)',
        'use_tent': True,
        'batch_size': 1,
        'tent_lr': LR,
        'tent_steps': 1,
        'episodic': False,
    },
]