import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Edges2ShoesDataset(Dataset):
    def __init__(self, root_dir, phase="train", image_size=512):
        self.phase = phase

        self.data_dir = os.path.join(root_dir, phase)
        self.image_paths = [
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ]

        self.image_size = image_size

        # transforms for target image (shoe)
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # transforms for condition (edge)
        self.cond_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()  
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        filename = os.path.basename(path)

        image = Image.open(path).convert("RGB")
        image = np.array(image)

        h, w, _ = image.shape

        # split image
        edge = image[:, :w // 2]
        shoe = image[:, w // 2:]

        edge = Image.fromarray(edge)
        shoe = Image.fromarray(shoe)

        # apply transforms
        edge = self.cond_transform(edge)
        shoe = self.image_transform(shoe)

        return {
            "shoe": shoe,
            "edge": edge
        }