import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x