import os
import sys
import argparse
import yaml
from datetime import datetime
import torch

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_smoke_test():
    """Run smoke test with reduced parameters for quick validation."""
    print("=" * 60)
    print("ZLA-LR-VAE SMOKE TEST")
    print("=" * 60)
    print(f"Smoke test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        config = load_config("config/smoke_test.yaml")
        print(f"Loaded smoke test configuration: {config['experiment']['name']}")
        print(f"Dataset: {config['experiment']['dataset']}")
        print(f"Epochs: {config['experiment']['epochs']}")
        print(f"Subset size: {config['experiment']['subset_size']}")
        print()
        
        print("Step 1: Data Loading and Preprocessing")
        print("-" * 40)
        from preprocess import load_dataset_by_name, create_data_loaders
        
        train_dataset, test_dataset = load_dataset_by_name(
            config['experiment']['dataset'], 
            subset_size=config['experiment']['subset_size']
        )
        train_loader, test_loader = create_data_loaders(
            train_dataset, test_dataset, 
            batch_size=config['experiment']['batch_size']
        )
        print("✓ Data loading completed successfully!")
        print()
        
        print("Step 2: VAE Model Training")
        print("-" * 40)
        from train import VAE, train_vae, save_vae_model, plot_training_curves
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = VAE(
            latent_dim=config['model']['latent_dim'],
            hidden_dim=config['model']['hidden_dim']
        )
        
        train_losses = train_vae(
            model, train_loader, 
            num_epochs=config['experiment']['epochs'],
            learning_rate=float(config['training']['learning_rate']),
            device=device
        )
        
        save_vae_model(model)
        plot_training_curves(train_losses, save_path=config['output']['images_dir'])
        print("✓ VAE training completed successfully!")
        print()
        
        print("Step 3: ZLA-LR-VAE Evaluation")
        print("-" * 40)
        from evaluate import evaluate_anomaly_detection, plot_score_distributions, save_results_json
        from preprocess import load_ood_datasets
        
        ood_loaders = load_ood_datasets(["cifar100"])
        
        results = evaluate_anomaly_detection(
            model, test_loader, ood_loaders, config, device
        )
        
        plot_score_distributions(results, save_path=config['output']['images_dir'])
        json_results = save_results_json(results, config, save_path=config['output']['results_dir'])
        print("✓ ZLA-LR-VAE evaluation completed successfully!")
        print()
        
        print("=" * 60)
        print("SMOKE TEST SUMMARY")
        print("=" * 60)
        print(f"Method: ZLA-LR-VAE")
        print(f"Dataset: {config['experiment']['dataset']}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Latent dimension: {config['model']['latent_dim']}")
        print(f"Hidden dimension: {config['model']['hidden_dim']}")
        print(f"Device used: {device}")
        print(f"Training epochs: {config['experiment']['epochs']}")
        print(f"Final training loss: {train_losses[-1]:.4f}")
        print(f"Tau threshold: {config['zla_lr']['tau']}")
        print(f"Max refinement steps: {config['zla_lr']['max_refinement_steps']}")
        print()
        
        print("Anomaly Detection Results:")
        for ood_name, ood_data in json_results['ood_results'].items():
            print(f"- {ood_name}: AUROC = {ood_data['auroc']:.4f}, AUPR = {ood_data['aupr']:.4f}")
        print()
        
        print("Generated Files:")
        print(f"- VAE training curves: {config['output']['images_dir']}/vae_training_curves.pdf")
        print(f"- Score distributions: {config['output']['images_dir']}/score_distributions.pdf")
        print(f"- Results JSON: {config['output']['results_dir']}/{config['experiment']['name']}_results.json")
        print("- Trained VAE model: models/zla_lr_vae_model.pth")
        print()
        
        print(f"Smoke test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during smoke test: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def run_full_experiment():
    """Run full experiment with complete parameters."""
    print("=" * 60)
    print("ZLA-LR-VAE FULL EXPERIMENT")
    print("=" * 60)
    print(f"Full experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        config = load_config("config/full_experiment.yaml")
        print(f"Loaded full experiment configuration: {config['experiment']['name']}")
        print(f"Dataset: {config['experiment']['dataset']}")
        print(f"Epochs: {config['experiment']['epochs']}")
        print()
        
        print("Step 1: Data Loading and Preprocessing")
        print("-" * 40)
        from preprocess import load_dataset_by_name, create_data_loaders
        
        train_dataset, test_dataset = load_dataset_by_name(config['experiment']['dataset'])
        train_loader, test_loader = create_data_loaders(
            train_dataset, test_dataset, 
            batch_size=config['experiment']['batch_size']
        )
        print("✓ Data loading completed successfully!")
        print()
        
        print("Step 2: VAE Model Training")
        print("-" * 40)
        from train import VAE, train_vae, save_vae_model, plot_training_curves
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = VAE(
            latent_dim=config['model']['latent_dim'],
            hidden_dim=config['model']['hidden_dim']
        )
        
        train_losses = train_vae(
            model, train_loader, 
            num_epochs=config['experiment']['epochs'],
            learning_rate=float(config['training']['learning_rate']),
            device=device
        )
        
        save_vae_model(model)
        plot_training_curves(train_losses, save_path=config['output']['images_dir'])
        print("✓ VAE training completed successfully!")
        print()
        
        print("Step 3: ZLA-LR-VAE Evaluation")
        print("-" * 40)
        from evaluate import evaluate_anomaly_detection, plot_score_distributions, save_results_json
        from preprocess import load_ood_datasets
        
        ood_loaders = load_ood_datasets(config['evaluation']['ood_datasets'])
        
        results = evaluate_anomaly_detection(
            model, test_loader, ood_loaders, config, device
        )
        
        plot_score_distributions(results, save_path=config['output']['images_dir'])
        json_results = save_results_json(results, config, save_path=config['output']['results_dir'])
        print("✓ ZLA-LR-VAE evaluation completed successfully!")
        print()
        
        print("=" * 60)
        print("FULL EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Method: ZLA-LR-VAE")
        print(f"Dataset: {config['experiment']['dataset']}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Latent dimension: {config['model']['latent_dim']}")
        print(f"Hidden dimension: {config['model']['hidden_dim']}")
        print(f"Device used: {device}")
        print(f"Training epochs: {config['experiment']['epochs']}")
        print(f"Final training loss: {train_losses[-1]:.4f}")
        print(f"Tau threshold: {config['zla_lr']['tau']}")
        print(f"Max refinement steps: {config['zla_lr']['max_refinement_steps']}")
        print()
        
        print("Anomaly Detection Results:")
        for ood_name, ood_data in json_results['ood_results'].items():
            print(f"- {ood_name}: AUROC = {ood_data['auroc']:.4f}, AUPR = {ood_data['aupr']:.4f}")
        print()
        
        print("Generated Files:")
        print(f"- VAE training curves: {config['output']['images_dir']}/vae_training_curves.pdf")
        print(f"- Score distributions: {config['output']['images_dir']}/score_distributions.pdf")
        print(f"- Results JSON: {config['output']['results_dir']}/{config['experiment']['name']}_results.json")
        print("- Trained VAE model: models/zla_lr_vae_model.pth")
        print()
        
        print(f"Full experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during full experiment: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point with command-line argument support."""
    parser = argparse.ArgumentParser(description='ZLA-LR-VAE Experimental Framework')
    parser.add_argument('--smoke-test', action='store_true', 
                       help='Run smoke test with reduced parameters for quick validation')
    parser.add_argument('--full-experiment', action='store_true',
                       help='Run full experiment with complete parameters')
    
    args = parser.parse_args()
    
    if args.smoke_test and args.full_experiment:
        print("Error: Cannot run both smoke test and full experiment simultaneously")
        return False
    
    if args.smoke_test:
        print("Running smoke test first...")
        smoke_success = run_smoke_test()
        if not smoke_success:
            print("Smoke test failed. Aborting.")
            return False
        print("Smoke test passed! Ready for full experiment.")
        return True
        
    elif args.full_experiment:
        print("Running full experiment...")
        return run_full_experiment()
        
    else:
        print("Error: Must specify either --smoke-test or --full-experiment")
        parser.print_help()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
