import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from train import ExampleMethodModel

def load_test_data(data_dir="data"):
    """Load preprocessed test data."""
    print(f"Loading test data from {data_dir}/...")
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    return X_test, y_test

def load_model(model_path="models/example_method_model.pth", input_dim=20):
    """Load the trained model."""
    print(f"Loading model from {model_path}...")
    model = ExampleMethodModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def evaluate_model(model, X_test, y_test, device='cpu'):
    """Evaluate the model on test data."""
    print("Evaluating model on test data...")
    
    model.to(device)
    model.eval()
    
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()
    
    accuracy = accuracy_score(y_test, predicted)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, predicted))
    
    return predicted, accuracy

def plot_confusion_matrix(y_true, y_pred, save_path=".research/iteration1/images"):
    """Plot and save confusion matrix as PDF."""
    os.makedirs(save_path, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix - example_method', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confusion_matrix.pdf"), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}/confusion_matrix.pdf")

def plot_prediction_distribution(y_true, y_pred, save_path=".research/iteration1/images"):
    """Plot prediction distribution as PDF."""
    os.makedirs(save_path, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(y_true, bins=2, alpha=0.7, label='True Labels', color='blue', edgecolor='black')
    ax1.set_title('True Label Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(y_pred, bins=2, alpha=0.7, label='Predicted Labels', color='red', edgecolor='black')
    ax2.set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "prediction_distribution.pdf"), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction distribution saved to {save_path}/prediction_distribution.pdf")

if __name__ == "__main__":
    print("=== Model Evaluation for example_method ===")
    print('Hello, world!')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_test, y_test = load_test_data()
    model = load_model(input_dim=X_test.shape[1])
    
    y_pred, accuracy = evaluate_model(model, X_test, y_test, device=device)
    
    plot_confusion_matrix(y_test, y_pred)
    plot_prediction_distribution(y_test, y_pred)
    
    print(f"\n=== Final Results ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Model evaluation completed successfully!")
