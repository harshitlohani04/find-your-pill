import torch
import torch.nn as nn

# Initial input layer
class InitialLayer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
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

# DenseLayer
class DenseLayer(nn.Module):
    def __init__(self, input_dim: int, growth_rate: int):
        super().__init__()
        self.input_dim = input_dim
        self.growth_rate = growth_rate

        self.block = nn.Sequential(
            nn.BatchNorm2d(self.input_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.input_dim,
                out_channels=4 * self.growth_rate,
                kernel_size=1,
                padding=0
            ),
            nn.BatchNorm2d(4*self.growth_rate),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=4 * self.growth_rate,
                out_channels=self.growth_rate,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):
        out = self.block(x)
        out = torch.cat([x, out], dim=1)
        return out
    
class DenseBlock(nn.Module):
    def __init__(self, input_dims: int, num_layers: int, growth_rate: int):
        super().__init__()
        channels = input_dims
        self.growth_rate = growth_rate
        self.num_layers = num_layers

        self.denseblock = nn.ModuleList()
        for _ in range(self.num_layers):
            self.denseblock.append(DenseLayer(input_dims=channels, growth_rate=self.growth_rate))
            channels += growth_rate
        self.out_channels = channels

    # forward path
    def forward(self, x):
        for layers in self.denseblock:
            x = layers(x)
        return x
    
# Transition Layer
class TransitionLayer(nn.Module):
    def __init__(self, input_dim: int, compression_factor: float = 0.5):
        super().__init__()

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
        super().__init__()
        self.in_channels = input_dim
        self.growth_rate = growth_rate
        self.number_of_blocks = number_of_blocks
        self.num_classes = num_classes

        self.initial_layer = InitialLayer(self.in_channels)
        channels = 64

        # dense + transition block logic
        self.dense_connections = nn.ModuleList()
        self.transition = nn.ModuleList()

        for i in range(self.number_of_blocks-1):
            numOfLayers = 6*(2**i)
            dense = DenseBlock(input_dims=channels, num_layers=numOfLayers, growth_rate=self.growth_rate)
            self.dense_connections.append(dense)
            channels += numOfLayers * (self.growth_rate)

            transition = TransitionLayer(input_dim=channels)
            self.transition.append(transition)
            channels = channels//2

        self.final_dense_block = DenseBlock(input_dims=channels, num_layers=16, growth_rate=self.growth_rate)
        channels += 16*self.growth_rate
        # final batch norm and other layers
        self.final_batch_norm = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(channels, self.num_classes)
        )

    def forward(self, x):
        x = self.initial_layer(x)

        # Processing through the dense and transtion blocks
        for dense, transition in zip(self.dense_connections, self.transition):
            x = dense(x)
            x = transition(x)
        # processing through the last layers
        x = self.final_dense_block(x)
        x = self.final_batch_norm(x)
        x = self.classification(x)

        return x