import torch 
from PIL import Image
import albumentations
import numpy as np
from color_constancy import color_constancy
import os

class MelonamaDataset:
    def __init__(self, image_paths, targets, augmentations=None, cc=False):
        self.image_paths = image_paths
        self.targets = targets
        self.augmentations = augmentations
        self.cc = cc

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path))
        target = self.targets[idx]

        # color constancy if not preprocessed          
        if self.cc: 
            image = color_constancy(image)
        
        # Image augmentations
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)      

        return {
            'image': torch.tensor(image, dtype=torch.float), 
            'target': torch.tensor(target, dtype=torch.long)
        } 


class MelonamaTTADataset:
    """Only useful for TTA during evaluation"""
    def __init__(self, image_paths, augmentations=None, nc=None):
        self.image_paths = image_paths
        self.augmentations = augmentations 
        self.nc = nc
        
    def __len__(self): return len(self.image_paths)
    
    def __getitem__(self, idx):
        # dummy targets
        target = torch.zeros(5 if self.nc==5 else 10)
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = self.augmentations(image)
        
        return {
            'image':image, 
            'target':target
        }