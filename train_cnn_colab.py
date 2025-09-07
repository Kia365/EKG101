"""
ECG CNN Training Script for Google Colab
This script is optimized for training on Google Colab with GPU support.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Import the CNN classes
from EKG_CNN import PTBCNNDataLoader, ECGCNNModel, plot_training_history, configure_gpu

def download_ptb_dataset():
    """Download PTB dataset if not available"""
    print("Checking for PTB dataset...")
    
    # Mount Google Drive first
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully!")
    except:
        print("Google Drive mounting failed or not in Colab environment")
    
    # Common paths where the dataset might be located (Google Drive first)
    possible_paths = [
        '/content/drive/MyDrive/ptb-diagnostic-ecg-database-1.0.0',
        '/content/drive/MyDrive/Downloads/ptb-diagnostic-ecg-database-1.0.0',
        '/content/drive/MyDrive/ECG_Project/ptb-diagnostic-ecg-database-1.0.0',
        '/content/ptb-diagnostic-ecg-database-1.0.0',
        os.path.expanduser('~/Downloads/ptb-diagnostic-ecg-database-1.0.0'),
        './ptb-diagnostic-ecg-database-1.0.0'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found PTB dataset at: {path}")
            return path
    
    print("PTB dataset not found. Please upload the dataset to your Google Drive.")
    print("You can download it from: https://physionet.org/content/ptb-db/1.0.0/")
    print("Upload it to one of these Google Drive locations:")
    print("  - /content/drive/MyDrive/ptb-diagnostic-ecg-database-1.0.0/")
    print("  - /content/drive/MyDrive/Downloads/ptb-diagnostic-ecg-database-1.0.0/")
    print("  - /content/drive/MyDrive/ECG_Project/ptb-diagnostic-ecg-database-1.0.0/")
    return None

def create_synthetic_data(n_samples=1000, signal_length=5000):
    """Create synthetic ECG data for testing when real data is not available"""
    print("Creating synthetic ECG data for testing...")
    
    np.random.seed(42)
    signals = []
    labels = []
    
    # Create different types of synthetic ECG patterns
    for i in range(n_samples):
        # Generate base signal
        t = np.linspace(0, 5, signal_length)  # 5 seconds at 1000 Hz
        
        # Normal ECG pattern
        if i < n_samples // 3:
            # Normal sinus rhythm
            heart_rate = np.random.uniform(60, 100)
            signal = np.sin(2 * np.pi * heart_rate / 60 * t) * 0.5
            signal += np.sin(2 * np.pi * heart_rate / 60 * 2 * t) * 0.1  # P wave
            signal += np.sin(2 * np.pi * heart_rate / 60 * 0.5 * t) * 0.2  # T wave
            label = "Normal"
            
        elif i < 2 * n_samples // 3:
            # Tachycardia
            heart_rate = np.random.uniform(100, 150)
            signal = np.sin(2 * np.pi * heart_rate / 60 * t) * 0.6
            signal += np.sin(2 * np.pi * heart_rate / 60 * 2 * t) * 0.15
            signal += np.sin(2 * np.pi * heart_rate / 60 * 0.5 * t) * 0.25
            label = "Tachycardia"
            
        else:
            # Bradycardia
            heart_rate = np.random.uniform(40, 60)
            signal = np.sin(2 * np.pi * heart_rate / 60 * t) * 0.4
            signal += np.sin(2 * np.pi * heart_rate / 60 * 2 * t) * 0.1
            signal += np.sin(2 * np.pi * heart_rate / 60 * 0.5 * t) * 0.15
            label = "Bradycardia"
        
        # Add noise
        noise = np.random.normal(0, 0.05, signal_length)
        signal += noise
        
        # Add baseline drift
        baseline = np.sin(2 * np.pi * 0.1 * t) * 0.1
        signal += baseline
        
        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        signals.append(signal)
        labels.append(label)
    
    signals = np.array(signals)
    signals = signals.reshape(signals.shape[0], signals.shape[1], 1)
    
    print(f"Created {len(signals)} synthetic ECG signals")
    print(f"Signal shape: {signals.shape}")
    print(f"Labels: {np.unique(labels)}")
    
    return signals, np.array(labels)

def main():
    """Main training function for Google Colab"""
    print("=" * 60)
    print("ECG CNN Training for Google Colab")
    print("=" * 60)
    
    # Configure GPU
    configure_gpu()
    
    # Try to find PTB dataset
    data_path = download_ptb_dataset()
    
    if data_path is not None:
        # Use real PTB data
        print("Using real PTB dataset...")
        loader = PTBCNNDataLoader(data_path=data_path)
        X, y, diagnoses = loader.load_all_records(max_records=200)
        
        if len(X) == 0:
            print("Failed to load PTB data. Creating synthetic data...")
            X, y = create_synthetic_data()
            diagnoses = y.tolist()
    else:
        # Use synthetic data
        print("Using synthetic ECG data...")
        X, y = create_synthetic_data()
        diagnoses = y.tolist()
    
    print(f"\nDataset Summary:")
    print(f"Total samples: {len(X)}")
    print(f"Signal shape: {X.shape}")
    print(f"Unique diagnoses: {np.unique(diagnoses)}")
    print(f"Class distribution:")
    for diagnosis in np.unique(diagnoses):
        count = np.sum(np.array(diagnoses) == diagnosis)
        print(f"  {diagnosis}: {count} samples")
    
    if len(X) > 10:
        # Create and train CNN model
        model = ECGCNNModel((X.shape[1], X.shape[2]), len(np.unique(y)))
        
        # Preprocess data
        X_processed, y_encoded = model.preprocess_data(X, y)
        
        print(f"\nModel Configuration:")
        print(f"Input shape: {model.input_shape}")
        print(f"Number of classes: {len(np.unique(y_encoded))}")
        print(f"Model parameters: {model.model.count_params():,}")
        
        # Check class distribution
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        print("\nClass distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {cls}: {count} samples")
        
        # Remove classes with too few samples
        min_samples = 3
        classes_to_remove = [cls for cls, count in zip(unique_classes, class_counts) if count < min_samples]
        
        if classes_to_remove:
            print(f"Removing classes with less than {min_samples} samples: {classes_to_remove}")
            valid_indices = [i for i, label in enumerate(y_encoded) if label not in classes_to_remove]
            X_processed = X_processed[valid_indices]
            y_encoded = y_encoded[valid_indices]
            print(f"After filtering: {len(X_processed)} samples")
        
        if len(X_processed) < 10 or len(np.unique(y_encoded)) < 2:
            print("Not enough data for training after filtering.")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\nData Split:")
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Train model
        print("\n" + "=" * 40)
        print("Starting CNN Training...")
        print("=" * 40)
        
        history = model.train(X_train, y_train, X_test, y_test, epochs=50, batch_size=32)
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate model
        print("\n" + "=" * 40)
        print("Model Evaluation")
        print("=" * 40)
        
        y_pred = model.evaluate(X_test, y_test)
        
        # Sample predictions
        print("\nSample Predictions:")
        print("-" * 30)
        for i in range(min(10, len(X_test))):
            signal = X_test[i]
            true_label = model.le.inverse_transform([y_test[i]])[0]
            pred_label, confidence, _ = model.predict_diagnosis(signal)
            print(f"Sample {i+1}: True={true_label}, Pred={pred_label}, Conf={confidence:.3f}")
        
        print("\n" + "=" * 40)
        print("Training Completed Successfully!")
        print("=" * 40)
        
        # Save model
        model.model.save('ptb_ecg_cnn_model.keras')
        print("CNN model saved as 'ptb_ecg_cnn_model.keras'")
        
        # Save to Google Drive if available
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            model.model.save('/content/drive/MyDrive/ptb_ecg_cnn_model.keras')
            print("Model also saved to Google Drive")
        except:
            print("Google Drive not available, model saved locally")
        
        # Save label encoder
        import joblib
        joblib.dump(model.le, 'cnn_label_encoder.pkl')
        print("Label encoder saved as 'cnn_label_encoder.pkl'")
        
        # Model summary
        print("\nModel Architecture Summary:")
        model.model.summary()
        
    else:
        print("Not enough data for training. Need at least 10 samples.")

if __name__ == "__main__":
    main()
