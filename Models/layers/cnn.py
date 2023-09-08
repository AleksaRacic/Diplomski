from torch import nn, Tensor

class CNN(nn.Module):
    def __init__(self, num_channels=32):
        
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)