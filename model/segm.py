import torch 
import torch.nn as nn


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1), # output the FCN channels
        ]
        super().__init__(*layers)