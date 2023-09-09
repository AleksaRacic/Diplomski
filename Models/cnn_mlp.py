import torch

from .layers.cnn import CNN

class CNN_MLP(torch.nn.Module):
    def __init__(self):
        super(CNN_MLP, self).__init__()
        self.cnn = CNN(input_channels=16)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(32 * 9 * 9, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 8)
        )
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, num_panels, height, width = x.size()
        x = self.cnn(x)
        x = x.view(-1, 32 * 9 * 9)
        x = self.fc(x)
        return torch.softmax(x, dim = 1)