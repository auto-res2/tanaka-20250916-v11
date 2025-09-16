import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class VAEEncoder(nn.Module):
    """Convolutional VAE encoder for 32x32 images."""
    
    def __init__(self, latent_dim=64, hidden_dim=128):
        super(VAEEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, hidden_dim//4, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//4, hidden_dim//2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU()
        )
        
        conv_output_size = hidden_dim * 2 * 2
        self.fc_mu = nn.Linear(conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_output_size, latent_dim)
    
    def forward(self, x):
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class VAEDecoder(nn.Module):
    """Convolutional VAE decoder for 32x32 images."""
    
    def __init__(self, latent_dim=64, hidden_dim=128):
        super(VAEDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.fc = nn.Linear(latent_dim, hidden_dim * 2 * 2)
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim//2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim//4, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), self.hidden_dim, 2, 2)
        return self.deconv_layers(h)

class VAE(nn.Module):
    """Variational Autoencoder for ZLA-LR-VAE."""
    
    def __init__(self, latent_dim=64, hidden_dim=128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim, hidden_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

def elbo_loss(recon_x, x, mu, logvar, beta=1.0):
    """Compute ELBO loss for VAE."""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss

def train_vae(model, train_loader, num_epochs=50, learning_rate=3e-4, device='cpu'):
    """Train the VAE model."""
    print(f"Training VAE for {num_epochs} epochs on {device}...")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    model.to(device)
    model.train()
    
    train_losses = []
    
    for epoch in tqdm(range(num_epochs), desc="Training VAE"):
        epoch_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = elbo_loss(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], ELBO Loss: {avg_loss:.4f}")
    
    return train_losses

def save_vae_model(model, model_path="models/zla_lr_vae_model.pth"):
    """Save the trained VAE model."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"VAE model saved to {model_path}")

def plot_training_curves(train_losses, save_path=".research/iteration3/images"):
    """Plot and save training curves as PDF."""
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('VAE Training Loss (ELBO)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('ELBO Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "vae_training_curves.pdf"), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}/vae_training_curves.pdf")

if __name__ == "__main__":
    print("=== VAE Training for ZLA-LR-VAE ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    from preprocess import load_dataset_by_name, create_data_loaders
    
    train_dataset, test_dataset = load_dataset_by_name("fashion_mnist", subset_size=1000)
    train_loader, _ = create_data_loaders(train_dataset, test_dataset)
    
    model = VAE(latent_dim=32, hidden_dim=64)
    train_losses = train_vae(model, train_loader, num_epochs=2, device=device)
    
    save_vae_model(model)
    plot_training_curves(train_losses)
    
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print("VAE training completed successfully!")
