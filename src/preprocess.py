import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
import os

def load_dataset_by_name(dataset_name, subset_size=None):
    """Load dataset by name with optional subset for smoke testing."""
    print(f"Loading dataset: {dataset_name}")
    
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
    elif dataset_name == "fashion_mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform
        )
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    if subset_size is not None:
        print(f"Using subset of {subset_size} samples for smoke test")
        train_indices = torch.randperm(len(train_dataset))[:subset_size]
        test_indices = torch.randperm(len(test_dataset))[:subset_size//5]
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset

def create_data_loaders(train_dataset, test_dataset, batch_size=64):
    """Create data loaders for training and testing."""
    train_loader = None
    test_loader = None
    
    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
    
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
    
    return train_loader, test_loader

def load_ood_datasets(dataset_names):
    """Load out-of-distribution datasets for evaluation."""
    ood_loaders = {}
    
    for dataset_name in dataset_names:
        print(f"Loading OOD dataset: {dataset_name}")
        
        if dataset_name == "svhn":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = torchvision.datasets.SVHN(
                root='./data', split='test', download=True, transform=transform
            )
            
        elif dataset_name == "cifar100":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            dataset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform
            )
            
        else:
            print(f"Skipping unsupported OOD dataset: {dataset_name}")
            continue
            
        ood_loaders[dataset_name] = DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=2
        )
    
    return ood_loaders

if __name__ == "__main__":
    print("=== Data Preprocessing for ZLA-LR-VAE ===")
    
    train_dataset, test_dataset = load_dataset_by_name("fashion_mnist", subset_size=1000)
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset)
    
    print("Data preprocessing completed successfully!")
