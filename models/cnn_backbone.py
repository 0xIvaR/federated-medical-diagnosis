import torch
import torch.nn as nn
from torchvision import models

class FrozenCNNBackbone(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True):
        super().__init__()
        
        if backbone_name == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 512
        elif backbone_name == 'mobilenet_v2':
            base_model = models.mobilenet_v2(pretrained=pretrained)
            self.features = base_model.features
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        for param in self.features.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, 1)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    backbone = FrozenCNNBackbone(backbone_name='resnet18').to(device)
    
    dummy_input = torch.randn(4, 3, 28, 28).to(device)
    output = backbone(dummy_input)
    
    print(f"✓ Backbone output shape: {output.shape}")
    print(f"✓ Feature dimension: {backbone.feature_dim}")
    
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    
    print(f"✓ Trainable params: {trainable:,}")
    print(f"✓ Total params: {total:,}")
    print(f"✓ Frozen: {100*(total-trainable)/total:.1f}%")