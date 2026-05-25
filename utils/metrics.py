import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

def calculate_metrics(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }

def get_classification_report(predictions, labels, class_names):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    return classification_report(labels, predictions, target_names=class_names)