import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json

from models.full_model import PathologyClassifier
from data.dataloader import get_data_loaders
from training.train import train_one_epoch, validate
from utils.energy_tracker import EnergyTracker

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    model = PathologyClassifier(num_classes=9).to(device)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)\n")
    
    train_loader, val_loader, test_loader, _ = get_data_loaders(batch_size=64)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.head.parameters(), lr=0.001)
    
    num_epochs = 20
    best_val_acc = 0.0
    
    results = {
        'train_history': [],
        'val_history': []
    }
    
    energy_tracker = EnergyTracker()
    energy_tracker.start()
    
    print("Starting training...\n")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        results['train_history'].append(train_metrics)
        results['val_history'].append(val_metrics)
        
        print(f"  Train Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"F1: {train_metrics['f1_macro']:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1: {val_metrics['f1_macro']:.4f}\n")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), 'experiments/best_model.pth')
            print(f"  ✓ New best model saved (Val Acc: {best_val_acc:.4f})\n")
    
    energy_stats = energy_tracker.stop()
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Training time: {energy_stats['time_minutes']:.2f} minutes")
    print(f"CO₂ emissions: {energy_stats['emissions_kg']:.6f} kg")
    
    model.load_state_dict(torch.load('experiments/best_model.pth', weights_only=True))
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    
    final_results = {
        'best_val_accuracy': best_val_acc,
        'test_metrics': test_metrics,
        'energy_stats': energy_stats,
        'training_history': results
    }
    
    with open('experiments/baseline_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\n✓ Results saved to experiments/baseline_results.json")

if __name__ == "__main__":
    main()