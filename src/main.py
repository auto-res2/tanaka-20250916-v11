import os
import sys
import time
from datetime import datetime

def main():
    """Main experimental pipeline for example_method."""
    print("=" * 60)
    print("EXAMPLE_METHOD EXPERIMENT - MAIN PIPELINE")
    print("=" * 60)
    print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('Hello, world!')
    print()
    
    try:
        print("Step 1: Data Preprocessing")
        print("-" * 30)
        from preprocess import generate_synthetic_data, preprocess_data, save_preprocessed_data
        
        X, y = generate_synthetic_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
        save_preprocessed_data(X_train, X_test, y_train, y_test, scaler)
        print("✓ Data preprocessing completed successfully!")
        print()
        
        print("Step 2: Model Training")
        print("-" * 30)
        from train import ExampleMethodModel, load_training_data, create_data_loader, train_model, save_model, plot_training_curves
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        X_train_loaded, y_train_loaded = load_training_data()
        train_loader = create_data_loader(X_train_loaded, y_train_loaded)
        
        model = ExampleMethodModel(input_dim=X_train_loaded.shape[1])
        train_losses, train_accuracies = train_model(model, train_loader, num_epochs=30, device=device)
        
        save_model(model)
        plot_training_curves(train_losses, train_accuracies)
        print("✓ Model training completed successfully!")
        print()
        
        print("Step 3: Model Evaluation")
        print("-" * 30)
        from evaluate import load_test_data, load_model, evaluate_model, plot_confusion_matrix, plot_prediction_distribution
        
        X_test_loaded, y_test_loaded = load_test_data()
        trained_model = load_model(input_dim=X_test_loaded.shape[1])
        
        y_pred, test_accuracy = evaluate_model(trained_model, X_test_loaded, y_test_loaded, device=device)
        
        plot_confusion_matrix(y_test_loaded, y_pred)
        plot_prediction_distribution(y_test_loaded, y_pred)
        print("✓ Model evaluation completed successfully!")
        print()
        
        print("=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        print(f"Method: example_method")
        print(f"Dataset: Synthetic classification data")
        print(f"Training samples: {len(X_train_loaded)}")
        print(f"Test samples: {len(X_test_loaded)}")
        print(f"Features: {X_train_loaded.shape[1]}")
        print(f"Classes: 2")
        print(f"Device used: {device}")
        print(f"Training epochs: 30")
        print(f"Final training accuracy: {train_accuracies[-1]:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print()
        
        print("Generated Files:")
        print("- Training curves: .research/iteration1/images/training_curves.pdf")
        print("- Confusion matrix: .research/iteration1/images/confusion_matrix.pdf")
        print("- Prediction distribution: .research/iteration1/images/prediction_distribution.pdf")
        print("- Trained model: models/example_method_model.pth")
        print("- Preprocessed data: data/")
        print()
        
        status_enum = "stopped"
        print(f"Experiment status: {status_enum}")
        print(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during experiment: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        status_enum = "stopped"
        print(f"Experiment status: {status_enum}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
