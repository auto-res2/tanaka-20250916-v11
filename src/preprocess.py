import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def generate_synthetic_data(n_samples=1000, n_features=20, n_classes=2, random_state=42):
    """Generate synthetic classification data for the example_method experiment."""
    print(f"Generating synthetic dataset with {n_samples} samples, {n_features} features, {n_classes} classes")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features//2,
        n_redundant=n_features//4,
        n_classes=n_classes,
        random_state=random_state
    )
    
    return X, y

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """Preprocess the data by splitting and scaling."""
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print(f"Class distribution in training set: {np.bincount(y_train)}")
    print(f"Class distribution in test set: {np.bincount(y_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_preprocessed_data(X_train, X_test, y_train, y_test, scaler, data_dir="data"):
    """Save preprocessed data to files."""
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Saving preprocessed data to {data_dir}/...")
    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)
    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)
    
    import joblib
    joblib.dump(scaler, os.path.join(data_dir, "scaler.pkl"))
    
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    print("=== Data Preprocessing for example_method ===")
    print('Hello, world!')
    
    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    save_preprocessed_data(X_train, X_test, y_train, y_test, scaler)
