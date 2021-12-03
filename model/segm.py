import torch 
import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def deconv4x4(in_planes:int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=4,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class TaskSpecificSubnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.seg_map_conv = nn.Sequential(
            conv3x3(1, 16, stride=5),
            conv3x3(16, 32, stride=4 ),
            conv3x3(32, 64, stride=2)
        )

        self.task_encoder = nn.Sequential(
            nn.Linear(7, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 676)
        )

        self.combine_rep_conv = nn.Sequential(
            conv3x3(128, 128, stride=1), 
            conv3x3(128, 128, stride=1), 
            deconv4x4(128, 32, stride=2),  
            deconv4x4(32, 16, stride=2),
            deconv4x4(16, 16, stride=2),        
            deconv4x4(16, 1, stride=5, dilation=4)    
        )

    def forward(self, seg_map_batch, task_label_batch):        
        seg_map_feats = self.seg_map_conv(seg_map_batch) 
                      
        task_label_batch = torch.stack([ x.reshape(26,26).unsqueeze(0) for x in self.task_encoder(task_label_batch) ]).expand(seg_map_feats.shape)
        class_stacked_representation = torch.hstack((seg_map_feats, task_label_batch))
        task_specific_attn_shift = self.combine_rep_conv(class_stacked_representation)       
        return task_specific_attn_shift   


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

