import torch
import torch.nn as nn
import os
import sys
from torchvision.models import resnet50
torch.set_grad_enabled(False);
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.logger import setup_logger
import logging

class DetrBackbone(nn.Module):
    def __init__(self, hidden_dim = 256):
        super().__init__()
        # creating resnet50 backbone
        self.backbone = resnet50()
        del self.backbone.fc
        
        # conv layer creation
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

    def forward(self, inputs):
        x = self.backbone.conv1(inputs)
        backbone_logger.debug(f"After conv1:   {x.shape}")
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        backbone_logger.debug(f"After maxpool: {x.shape}")
        x = self.backbone.layer1(x)
        backbone_logger.debug(f"After layer1:  {x.shape}")
        x = self.backbone.layer2(x)
        backbone_logger.debug(f"After layer2:  {x.shape}")
        x = self.backbone.layer3(x)
        backbone_logger.debug(f"After layer3:  {x.shape}")
        x = self.backbone.layer4(x)
        backbone_logger.debug(f"After layer4:  {x.shape}")
        h = self.conv(x)
        backbone_logger.debug(f"The final output:  {h.shape}")
        
        return h

if __name__ == "__main__":
    
    backbone_logger = setup_logger(name='backbone_results', level = logging.DEBUG)
    model = resnet50()
    backbone_logger.info('=' * 50)
    backbone_logger.info('The backbone model description is given below:')
    backbone_logger.info(model)
    backbone_logger.info('=' *50)
    backbone_logger.info('children name of resnet50 are given below:')
    for i, child in enumerate(model.children()):
        backbone_logger.info(f"Child {i}: {child.__class__.__name__}")
    

    backbone = DetrBackbone(hidden_dim = 256)
    dummy_input = torch.randn(1, 3, 800, 1200)
    features = backbone(dummy_input)
    backbone_logger.debug(f"Input shape:  {dummy_input.shape}")
