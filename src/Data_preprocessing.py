import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets, transforms

class DirectionalMNISTDataset(Dataset):
    def __init__(self, train=True):
        self.mnist = datasets.MNIST(
            root="./data", 
            train=train, 
            download=True, 
            transform=transforms.ToTensor()
        )
        
    def __len__(self):
        return len(self.mnist)
    
    def simulate_motion(self, image_tensor):
        img_np = image_tensor.squeeze().numpy()
        binary = (img_np > 0.5).astype(np.float32)
        dist_map = distance_transform_edt(binary)
        h, w = dist_map.shape
        y_coords = np.linspace(1.0, 0.0, h) # Linear gradient
        # Broadcast to (28, 28)
        vertical_gradient = np.tile(y_coords[:, None], (1, w))
        priority_map = dist_map * (vertical_gradient + 0.5) 
        # Mask out background (0 stays 0)
        priority_map = priority_map * binary
        frames = []
        max_val = priority_map.max() + 1e-5
        for i in range(16):
            thresh = (1.0 - (i / 15.0)) * max_val 
            frame = (priority_map > thresh).astype(np.float32)
            frames.append(frame)
        return torch.tensor(np.array(frames)).unsqueeze(0), img_np

    def __getitem__(self, idx):
        static_img, label = self.mnist[idx]
        video, raw_img = self.simulate_motion(static_img)
        
        # Normalize to [-1, 1]
        return (video * 2) - 1, label