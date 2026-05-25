import torch
import torch.nn as nn
from models.cnn_backbone import FrozenCNNBackbone
from models.lightweight_head import LightweightHead

class PathologyClassifier(nn.Module):
    def __init__(self, num_classes=9, backbone_name='resnet18'):
        super().__init__()
        
        self.backbone = FrozenCNNBackbone(backbone_name=backbone_name)
        self.head = LightweightHead(
            input_dim=self.backbone.feature_dim,
            num_classes=num_classes
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = PathologyClassifier(num_classes=9).to(device)
    
    dummy_input = torch.randn(8, 3, 28, 28).to(device)
    output = model(dummy_input)
    
    print(f"✓ Model output shape: {output.shape}")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    
    print(f"✓ Total params: {total:,}")
    print(f"✓ Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"✓ Frozen params: {frozen:,} ({100*frozen/total:.2f}%)")