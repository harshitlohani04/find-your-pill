import torch
import torch.nn as nn

# Initial input layer
class InitialLayer(nn.Module):
    def __init__(self, input_dim: int):
        super(InitialLayer, self).__init__()
        self.input_dim = input_dim

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=32,
                kernel_size=7,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

# DenseBlock
class DenseBlock(nn.Module):
    def __init__(self, input_dim: int, growth_rate: int, num_layers: int):
        super(DenseBlock, self).__init__()
        self.input_dim = input_dim
        self.growth_rate = growth_rate
        self.num_layers = num_layers

        layers = []

        for i in range(num_layers):
            layers.append(nn.BatchNorm2d(self.input_dim))
            layers.append(nn.ReLU())
            layers.append(
                nn.Conv2d(
                    in_channels=self.input_dim + i*self.growth_rate,
                    out_channels=4 * self.growth_rate,
                    kernel_size=1,
                    padding=0
                )
            )
            layers.append(nn.BatchNorm2d(4*self.growth_rate))
            layers.append(nn.ReLU())
            layers.append(
                nn.Conv2d(
                    in_channels=4 * self.growth_rate,
                    out_channels=self.growth_rate,
                    kernel_size=3,
                    padding=1
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
# Transition Layer
class TransitionLayer(nn.Module):
    def __init__(self, input_dim: int, compression_factor: float = 0.5):
        super(TransitionLayer, self).__init__()

        self.input_dim = input_dim
        self.theta = compression_factor

        self.transition = nn.Sequential(
            nn.BatchNorm2d(self.input_dim),
            nn.Conv2d(
                in_channels = self.input_dim,
                out_channels = self.theta * self.input_dim,
                kernel_size = 1,
                padding = 0
            ),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        x = self.transition(x)
        return x

# Final Dense Net

