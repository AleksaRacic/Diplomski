import cv2
import numpy as np
import torch

class ResizeImage:
    def __init__(self, img_size):
        self.img_size = img_size
    
    def __call__(self, image):
        resize_image = []
        for idx in range(0, 16):
            resize_image.append(
                cv2.resize(image[idx, :], (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST))
        return np.stack(resize_image)

class ToTensor:
    def __call__(self, sample):
        return torch.tensor(np.array(sample), dtype=torch.float32)
        
