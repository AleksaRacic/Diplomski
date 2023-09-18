import torch
import numpy as np

from glob import glob
from os import path

from torch.utils.data import Dataset
from torchvision.transforms import Compose


class PGM_dataset(Dataset):
    '''
        Dataset class for the PGM dataset
        
        root_folder: path to the folder containing the npz files
        img_size: size of the image to be resized to
        transform: transform to be applied to the image
        trian_mode: if True, the dataset will be shuffled
    '''

    def __init__(self, root_folder: str, transform:Compose=None):
        self.transform = transform
        self.file_names = glob(path.join(root_folder, '*.npz'))

    def __len__(self)->int:
        return len(self.file_names)

    def __getitem__(self, idx:int) -> torch.Tensor:
        data = np.load(self.file_names[idx])
        image = data['image'].reshape(16, 160, 160)
        target = data['target']

        del data

        if self.transform:
            transformed_image = self.transform(image)
            target = torch.tensor(target, dtype=torch.long)
        return transformed_image, target