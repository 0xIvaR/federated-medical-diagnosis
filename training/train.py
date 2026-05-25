import torch
import torch.nn as nn
from tqdm import tqdm
from utils.metrics import calculate_metrics

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    model.backbone.eval()
    
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).squeeze().long()
        
        optimizer.zero_grad()
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        predictions = torch.argmax(logits, dim=1)
        all_predictions.append(predictions)
        all_labels.append(labels)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / len(train_loader)
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    metrics = calculate_metrics(all_predictions, all_labels)
    metrics['loss'] = avg_loss
    
    return metrics

def validate(model, val_loader, criterion, device):
    model.eval()
    
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.to(device).squeeze().long()
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            running_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            all_predictions.append(predictions)
            all_labels.append(labels)
    
    avg_loss = running_loss / len(val_loader)
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    metrics = calculate_metrics(all_predictions, all_labels)
    metrics['loss'] = avg_loss
    
    return metrics