import timm
import torch.nn as nn 


class Encoder(nn.Module):

    def __init__(self, model_name="resnet50"):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0,  global_pool='')

    def forward(self, img_batch):
        backbone_feats = self.backbone(img_batch)
        return backbone_feats





