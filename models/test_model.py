import torch
from models.full_model import PathologyClassifier
from data.dataloader import get_data_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PathologyClassifier(num_classes=9).to(device)
model.eval()

train_loader, _, _, _ = get_data_loaders(batch_size=32)

images, labels = next(iter(train_loader))
images = images.to(device)
labels = labels.to(device)

with torch.no_grad():
    logits = model(images)
    predictions = torch.argmax(logits, dim=1)

print(f"✓ Input shape: {images.shape}")
print(f"✓ Output shape: {logits.shape}")
print(f"✓ Predictions shape: {predictions.shape}")
print(f"✓ Sample predictions: {predictions[:10].cpu().numpy()}")
print(f"✓ Sample labels: {labels[:10].flatten().cpu().numpy()}")