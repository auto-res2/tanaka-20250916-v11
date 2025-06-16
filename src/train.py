import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class ExampleMethodModel(nn.Module):
    """Simple neural network for the example_method experiment."""
    
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(ExampleMethodModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def load_training_data(data_dir="data"):
    """Load preprocessed training data."""
    print(f"Loading training data from {data_dir}/...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    return X_train, y_train

def create_data_loader(X, y, batch_size=32, shuffle=True):
    """Create PyTorch DataLoader from numpy arrays."""
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_model(model, train_loader, num_epochs=50, learning_rate=0.001, device='cpu'):
    """Train the example_method model."""
    print(f"Training model for {num_epochs} epochs on {device}...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    model.train()
    
    train_losses = []
    train_accuracies = []
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += batch_y.size(0)
            correct_predictions += (predicted == batch_y).sum().item()
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return train_losses, train_accuracies

def save_model(model, model_path="models/example_method_model.pth"):
    """Save the trained model."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def plot_training_curves(train_losses, train_accuracies, save_path=".research/iteration1/images"):
    """Plot and save training curves as PDF."""
    os.makedirs(save_path, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accuracies, 'r-', linewidth=2)
    ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_curves.pdf"), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}/training_curves.pdf")

if __name__ == "__main__":
    print("=== Model Training for example_method ===")
    print('Hello, world!')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    X_train, y_train = load_training_data()
    train_loader = create_data_loader(X_train, y_train)
    
    model = ExampleMethodModel(input_dim=X_train.shape[1])
    train_losses, train_accuracies = train_model(model, train_loader, device=device)
    
    save_model(model)
    plot_training_curves(train_losses, train_accuracies)
    
    print(f"Final training accuracy: {train_accuracies[-1]:.4f}")
    print("Model training completed successfully!")
