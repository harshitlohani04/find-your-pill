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
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
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

# Final Dense Net - implementation 121
class DenseNet121(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 3, growth_rate: int = 32, number_of_blocks: int = 4):
        super(DenseNet121).__init__()
        self.input_dim = input_dim
        self.growth_rate = growth_rate
        self.number_of_blocks = number_of_blocks
        self.num_classes = num_classes

        self.initial_layer = InitialLayer(input_dim)

    def forward(self, x):
        x = self.initial_layer(x)

        # Processing through the dense and transtion blocks
        for i in range(self.number_of_blocks - 1):
            x = DenseBlock(
                input_dim = x.shape[1],
                growth_rate = self.growth_rate,
                num_layers = 6 * (2**i)
            )(x)
            x = TransitionLayer(
                input_dim = x.shape[1]
            )(x)
        x = DenseBlock(
            input_dim = x.shape[1],
            growth_rate = self.growth_rate,
            num_layers = 16
        )(x)

        x = nn.Sequential(
            nn.BatchNorm2d(x.shape[1]),
            nn.ReLU()
        )(x)

        # Final classification layer
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = nn.Flatten()(x)
        x = nn.Linear(x.shape[1], self.num_classes)(x)

        return x