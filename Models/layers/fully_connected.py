from torch import nn, Tensor

class LinearBn(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor):
        x = self.linear(x)
        tensor_shape_before_bn = x.shape
        x = x.flatten(0, -2)
        x = self.bn(x)
        x = x.view(tensor_shape_before_bn)
        x = self.relu(x)
        return x


class DeepLinearLayerG(nn.Module):
    """Funkcija koja trazi relacije izmedju svakog para panela."""
    def __init__(self, in_channels: int, out_channels: int):
        super(DeepLinearLayerG, self).__init__()
        self.fc = nn.Sequential(
            LinearBn(in_channels, out_channels),
            LinearBn(out_channels, out_channels),
            LinearBn(out_channels, out_channels),
            LinearBn(out_channels, out_channels)
        )
    def forward(self, x: Tensor) ->Tensor:
        x = self.fc(x)
        x = x.sum(dim=1) # sumiramo vrednost po prvoj dimenziji TODO ovo odvoji u combiner feature
        return x

class DeepLinearLayerF(nn.Module):
    """Funkcija koja ocenjuje svaki choice panel"""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5):
        super(DeepLinearLayerF, self).__init__()
        self.fc = nn.Sequential(
            LinearBn(in_channels, 256),
            LinearBn(256, 256),
            #nn.Dropout(dropout),
            nn.Linear(256, out_channels)
        )
    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        return x