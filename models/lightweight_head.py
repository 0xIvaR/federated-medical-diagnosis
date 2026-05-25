import torch
import torch.nn as nn

class LightweightHead(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    head = LightweightHead(input_dim=512, num_classes=9).to(device)
    
    dummy_features = torch.randn(4, 512).to(device)
    output = head(dummy_features)
    
    print(f"✓ Head output shape: {output.shape}")
    
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"✓ Trainable params: {trainable:,}")