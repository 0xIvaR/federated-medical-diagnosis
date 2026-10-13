import torch
import pytest
from models.full_model import PathologyClassifier


def test_pathology_classifier_forward_pass():
    """Validates model initialization, forward pass shape, and output dimensionality."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PathologyClassifier(num_classes=9).to(device)
    model.eval()

    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 28, 28).to(device)
    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (batch_size, 9), f"Expected shape ({batch_size}, 9), got {logits.shape}"


def test_frozen_backbone_parameter_budget():
    """Validates that ResNet-18 backbone is completely frozen and head is trainable."""
    model = PathologyClassifier(num_classes=9)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params

    # Backbone is ResNet-18 (~11.17M params) and Head is 2-layer MLP (~133.6K params)
    assert trainable_params == 133641, f"Expected 133,641 trainable params, got {trainable_params}"
    assert frozen_params > 11000000, f"Expected >11M frozen params, got {frozen_params}"

    # Verify head parameters have gradients enabled, backbone does not
    for param in model.backbone.parameters():
        assert not param.requires_grad, "Backbone parameters must have requires_grad=False"

    for param in model.head.parameters():
        assert param.requires_grad, "Head parameters must have requires_grad=True"
