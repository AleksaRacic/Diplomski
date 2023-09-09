from torchvision.transforms import Compose
from transforms import ResizeImage, ToTensor

from data_loader import PGM_dataset

def get_transforms():
    return Compose([ResizeImage(80), ToTensor()])



