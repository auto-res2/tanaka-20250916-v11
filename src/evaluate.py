import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import json
import os
from train import VAE

def load_vae_model(model_path="models/zla_lr_vae_model.pth", latent_dim=64, hidden_dim=128):
    """Load the trained VAE model."""
    print(f"Loading VAE model from {model_path}...")
    model = VAE(latent_dim=latent_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def elbo(x, mu, logvar, vae_decoder, device='cpu'):
    """Compute ELBO for given posterior parameters."""
    z = torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu
    recon_x = vae_decoder(z)
    
    recon_loss = F.mse_loss(recon_x, x, reduction='none').sum(dim=[1,2,3])
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    return -(recon_loss + kld_loss)

def zla_lr_score(x, vae_model, tau=0.1, J=3, device='cpu'):
    """ZLA-LR core implementation for anomaly detection."""
    vae_model.eval()
    x = x.to(device)
    vae_model = vae_model.to(device)
    
    with torch.no_grad():
        mu_b, logv_b = vae_model.encode(x)
        sigma_b = torch.exp(0.5 * logv_b)
    
    mu_b = mu_b.detach().requires_grad_(True)
    recon_x = vae_model.decode(mu_b)
    recon_ll = -F.mse_loss(recon_x, x, reduction='none').sum(dim=[1,2,3])
    
    g = torch.autograd.grad(recon_ll.sum(), mu_b, retain_graph=False)[0]
    
    v = torch.randn_like(mu_b)
    mu_b_for_jvp = mu_b.detach().requires_grad_(True)
    Jv = torch.autograd.functional.jvp(
        lambda z: vae_model.decode(z), (mu_b_for_jvp,), (v,), create_graph=False
    )[1]
    
    obs_var = 1.0
    h_diag = (Jv.flatten(1).pow(2).mean(-1, keepdim=True) / obs_var + 1).expand_as(mu_b)
    
    mu_L = mu_b.detach() + g / h_diag
    sigma_L = h_diag.rsqrt()
    
    elbo_L = elbo(x, mu_L, 2*sigma_L.log(), vae_model.decoder, device)
    elbo_b = elbo(x, mu_b.detach(), logv_b, vae_model.decoder, device)
    lr_lap = elbo_L - elbo_b
    
    delta_hat = 0.5 * (g.pow(2) / h_diag).sum(dim=1)
    
    if delta_hat.mean().item() > tau and J > 0:
        mu_r = mu_L.clone().detach().requires_grad_()
        logv_r = (2*sigma_L.log()).clone().detach().requires_grad_()
        opt = torch.optim.SGD([mu_r, logv_r], lr=1e-3)
        
        for _ in range(J):
            loss = -elbo(x, mu_r, logv_r, vae_model.decoder, device).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        elbo_r = elbo(x, mu_r, logv_r, vae_model.decoder, device)
        return (elbo_r - elbo_b).cpu(), delta_hat.cpu()
    else:
        return lr_lap.cpu(), delta_hat.cpu()

def evaluate_anomaly_detection(vae_model, id_loader, ood_loaders, config, device='cpu'):
    """Evaluate ZLA-LR-VAE on anomaly detection task."""
    print("Evaluating ZLA-LR-VAE for anomaly detection...")
    
    tau = config['zla_lr']['tau']
    max_steps = config['zla_lr']['max_refinement_steps']
    
    id_scores = []
    id_deltas = []
    
    print("Computing scores for ID data...")
    for batch_idx, (data, _) in enumerate(id_loader):
        if batch_idx >= 10:
            break
        scores, deltas = zla_lr_score(data, vae_model, tau=tau, J=max_steps, device=device)
        id_scores.extend(scores.detach().numpy())
        id_deltas.extend(deltas.detach().numpy())
    
    results = {
        'id_scores': id_scores,
        'id_deltas': id_deltas,
        'ood_results': {}
    }
    
    for ood_name, ood_loader in ood_loaders.items():
        print(f"Computing scores for OOD data: {ood_name}")
        ood_scores = []
        ood_deltas = []
        
        for batch_idx, (data, _) in enumerate(ood_loader):
            if batch_idx >= 10:
                break
            scores, deltas = zla_lr_score(data, vae_model, tau=tau, J=max_steps, device=device)
            ood_scores.extend(scores.detach().numpy())
            ood_deltas.extend(deltas.detach().numpy())
        
        if len(ood_scores) > 0 and len(id_scores) > 0:
            y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
            y_scores = np.concatenate([id_scores, ood_scores])
            
            auroc = roc_auc_score(y_true, y_scores)
            aupr = average_precision_score(y_true, y_scores)
            
            results['ood_results'][ood_name] = {
                'auroc': float(auroc),
                'aupr': float(aupr),
                'ood_scores': ood_scores,
                'ood_deltas': ood_deltas
            }
            
            print(f"{ood_name} - AUROC: {auroc:.4f}, AUPR: {aupr:.4f}")
    
    return results

def plot_score_distributions(results, save_path=".research/iteration5/images"):
    """Plot score distributions for ID and OOD data."""
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    id_scores = results['id_scores']
    
    axes[0,0].hist(id_scores, bins=30, alpha=0.7, label='ID Scores', color='blue')
    axes[0,0].set_title('ID Score Distribution')
    axes[0,0].set_xlabel('ZLA-LR Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()
    
    axes[0,1].hist(results['id_deltas'], bins=30, alpha=0.7, label='ID Deltas', color='green')
    axes[0,1].set_title('ID Delta Distribution')
    axes[0,1].set_xlabel('Delta Hat')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    
    for idx, (ood_name, ood_data) in enumerate(results['ood_results'].items()):
        if idx >= 2:
            break
        
        ax = axes[1, idx]
        ax.hist(id_scores, bins=30, alpha=0.5, label='ID', color='blue')
        ax.hist(ood_data['ood_scores'], bins=30, alpha=0.5, label=f'OOD ({ood_name})', color='red')
        ax.set_title(f'Score Comparison: ID vs {ood_name}')
        ax.set_xlabel('ZLA-LR Score')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "score_distributions.pdf"), format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Score distributions saved to {save_path}/score_distributions.pdf")

def save_results_json(results, config, save_path=".research/iteration5"):
    """Save evaluation results as JSON."""
    os.makedirs(save_path, exist_ok=True)
    
    experiment_name = config['experiment']['name']
    results_file = os.path.join(save_path, f"{experiment_name}_results.json")
    
    json_results = {
        'experiment_name': experiment_name,
        'config': config,
        'num_id_samples': len(results['id_scores']),
        'mean_id_score': float(np.mean(results['id_scores'])),
        'std_id_score': float(np.std(results['id_scores'])),
        'mean_id_delta': float(np.mean(results['id_deltas'])),
        'ood_results': {}
    }
    
    for ood_name, ood_data in results['ood_results'].items():
        json_results['ood_results'][ood_name] = {
            'auroc': ood_data['auroc'],
            'aupr': ood_data['aupr'],
            'num_samples': len(ood_data['ood_scores']),
            'mean_score': float(np.mean(ood_data['ood_scores'])),
            'std_score': float(np.std(ood_data['ood_scores']))
        }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    print("JSON Results:")
    print(json.dumps(json_results, indent=2))
    
    return json_results

if __name__ == "__main__":
    print("=== ZLA-LR-VAE Evaluation ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    from preprocess import load_dataset_by_name, create_data_loaders, load_ood_datasets
    
    config = {
        'experiment': {'name': 'ZLA-LR-VAE-test'},
        'zla_lr': {'tau': 0.05, 'max_refinement_steps': 1}
    }
    
    train_dataset, test_dataset = load_dataset_by_name("fashion_mnist", subset_size=1000)
    _, test_loader = create_data_loaders(train_dataset, test_dataset)
    
    ood_loaders = load_ood_datasets(["cifar100"])
    
    vae_model = load_vae_model(latent_dim=32, hidden_dim=64)
    results = evaluate_anomaly_detection(vae_model, test_loader, ood_loaders, config, device)
    
    plot_score_distributions(results)
    save_results_json(results, config)
    
    print("ZLA-LR-VAE evaluation completed successfully!")
