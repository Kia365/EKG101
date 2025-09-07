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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

from EKG import PTBDataLoader, ECGFeatureModel, plot_training_history

def main():
    data_path = os.path.expanduser('~/Downloads/ptb-diagnostic-ecg-database-1.0.0')
    
    loader = PTBDataLoader(data_path=data_path)
    
    print("Loading PTB database with more records...")
    X, y, diagnoses = loader.load_all_records(max_records=50)
    
    print(f"Successfully loaded {len(X)} records")
    print(f"Unique diagnoses: {np.unique(diagnoses)}")
    print(f"Class distribution:")
    for diagnosis in np.unique(diagnoses):
        count = np.sum(np.array(diagnoses) == diagnosis)
        print(f"  {diagnosis}: {count} samples")
    
    if len(np.unique(diagnoses)) <= 1:
        print("\nOnly one class found. Creating synthetic diverse data...")
        n_samples = len(X)
        
        new_y = []
        for features in X:
            heart_rate = features[0]
            qt_interval = features[7]
            
            if heart_rate > 80 and qt_interval > 5:
                new_y.append("Tachycardia_with_LQT")
            elif heart_rate < 60 and qt_interval > 5:
                new_y.append("Bradycardia_with_LQT")
            elif heart_rate > 80:
                new_y.append("Tachycardia")
            elif heart_rate < 60:
                new_y.append("Bradycardia")
            elif qt_interval > 5:
                new_y.append("Long_QT")
            else:
                new_y.append("Normal_ECG")
        
        y = np.array(new_y)
        diagnoses = new_y
        
        print(f"Created synthetic labels. New class distribution:")
        for diagnosis in np.unique(diagnoses):
            count = np.sum(y == diagnosis)
            print(f"  {diagnosis}: {count} samples")
    
    if len(X) > 10:
        model = ECGFeatureModel((X.shape[1],), len(np.unique(y)))
        X_scaled, y_encoded = model.preprocess_data(X, y)
        
        print("\nChecking class distribution for stratification...")
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        
        classes_to_remove = []
        for cls, count in zip(unique_classes, class_counts):
            if count < 2:
                classes_to_remove.append(cls)
        
        if classes_to_remove:
            print(f"Removing classes with less than 2 samples: {classes_to_remove}")
            valid_indices = [i for i, label in enumerate(y_encoded) if label not in classes_to_remove]
            X_scaled = X_scaled[valid_indices]
            y_encoded = y_encoded[valid_indices]
            print(f"After filtering: {len(X_scaled)} samples")
        
        unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
        print("Final class distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {cls}: {count} samples")
        
        if len(X_scaled) < 10 or len(np.unique(y_encoded)) < 2:
            print("Not enough data for training after filtering.")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")
        
        print("Starting model training...")
        history = model.train(X_train, y_train, X_test, y_test, epochs=100)
        
        plot_training_history(history)
        
        print("Evaluating model...")
        y_pred = model.evaluate(X_test, y_test)
        
        print("\nSample predictions:")
        for i in range(min(5, len(X_test))):
            features = X_test[i]
            true_label = model.le.inverse_transform([y_test[i]])[0]
            pred_label, confidence, _ = model.predict_diagnosis(features)
            print(f"True: {true_label}, Pred: {pred_label}, Confidence: {confidence:.3f}")
        
        print("\nModel training completed successfully!")
        
        model.model.save('ptb_ecg_model.keras')
        print("Model saved as 'ptb_ecg_model.keras'")
        
        import joblib
        joblib.dump(model.scaler, 'scaler.pkl')
        joblib.dump(model.le, 'label_encoder.pkl')
        print("Preprocessing objects saved.")
        
    else:
        print("Not enough data for training. Need at least 10 samples.")

if __name__ == "__main__":
    main()
