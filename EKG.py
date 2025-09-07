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

class PTBDataLoader:
    def __init__(self, data_path=os.path.expanduser('~/Downloads/ptb-diagnostic-ecg-database-1.0.0')):
        self.data_path = data_path
        self.records = self.get_record_list()
    
    def get_record_list(self):
        print(f"Checking data path: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            print("Data path does not exist!")
            return []
        
        records = []
        
        for patient_folder in os.listdir(self.data_path):
            patient_path = os.path.join(self.data_path, patient_folder)
            
            if os.path.isdir(patient_path):
                print(f"Checking patient folder: {patient_folder}")
                
                for file in os.listdir(patient_path):
                    if file.endswith('.hea'):
                        record_name = file.replace('.hea', '')
                        record_full_path = os.path.join(patient_path, record_name)
                        
                        dat_file = record_full_path + '.dat'
                        if os.path.exists(dat_file):
                            records.append(record_full_path)
                            print(f"Found record: {record_full_path}")
        
        print(f"Total records found: {len(records)}")
        return records
    
    def load_record(self, record_full_path):
        try:
            print(f"Loading record: {record_full_path}")
            
            record = wfdb.rdrecord(record_full_path)
            print(f"Successfully loaded: {os.path.basename(record_full_path)}")
            print(f"Signal shape: {record.p_signal.shape}")
            print(f"Sampling frequency: {record.fs} Hz")
            print(f"Number of leads: {record.n_sig}")
            print(f"Signal duration: {len(record.p_signal)/record.fs:.1f} seconds")
            
            return record
        except Exception as e:
            print(f"Error loading record {record_full_path}: {str(e)}")
            return None
    
    def get_patient_info(self, record_full_path):
        try:
            header_path = record_full_path + '.hea'
            if os.path.exists(header_path):
                with open(header_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                return lines
            return []
        except Exception as e:
            print(f"Error reading header {record_full_path}: {e}")
            return []
    
    def extract_diagnosis(self, header_lines, record_path):
        diagnosis = "Unknown"
        
        for line in header_lines:
            line_lower = line.strip().lower()
            
            if 'myocardial' in line_lower or 'infarction' in line_lower or 'mi' in line_lower:
                diagnosis = "Myocardial Infarction"
                break
            elif 'cardiomyopathy' in line_lower:
                diagnosis = "Cardiomyopathy"
                break
            elif 'bundle' in line_lower or 'block' in line_lower or 'bbb' in line_lower:
                diagnosis = "Bundle Branch Block"
                break
            elif 'healthy' in line_lower or 'normal' in line_lower or 'control' in line_lower:
                diagnosis = "Healthy Control"
                break
            elif 'hypertrophy' in line_lower:
                diagnosis = "Ventricular Hypertrophy"
                break
            elif 'dysfunction' in line_lower:
                diagnosis = "Cardiac Dysfunction"
                break
            elif 'ischemia' in line_lower:
                diagnosis = "Myocardial Ischemia"
                break
            elif 'st elevation' in line_lower:
                diagnosis = "ST Elevation"
                break
            elif 't wave' in line_lower:
                diagnosis = "T Wave Abnormality"
                break
        
        if diagnosis == "Unknown":
            patient_folder = os.path.basename(os.path.dirname(record_path))
            
            if 'patient' in patient_folder.lower():
                try:
                    patient_num = int(patient_folder.replace('patient', '').strip())
                    if patient_num % 2 == 0:
                        diagnosis = "Cardiac Patient"
                    else:
                        diagnosis = "Healthy Control"
                except:
                    diagnosis = f"Patient_{patient_folder}"
        
        return diagnosis
    
    def extract_features_from_signal(self, signal, fs=1000):
        if signal is None or len(signal) < 100:
            return [0] * 12
        
        features = []
        
        try:
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            max_val = np.max(signal)
            min_val = np.min(signal)
            range_val = max_val - min_val
            
            threshold = np.mean(signal) + 2 * np.std(signal)
            peaks = self.find_peaks_simple(signal, threshold)
            
            if len(peaks) > 1:
                rr_intervals = np.diff(peaks) / fs
                heart_rate = 60 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
                hr_std = np.std(rr_intervals) if len(rr_intervals) > 1 else 0
            else:
                heart_rate = 0
                hr_std = 0
            
            fft_vals = np.fft.fft(signal)
            fft_magnitude = np.abs(fft_vals)
            
            if len(fft_magnitude) > 1:
                dominant_freq = np.argmax(fft_magnitude[1:len(fft_vals)//2]) + 1
            else:
                dominant_freq = 0
            
            from scipy import stats
            skewness = stats.skew(signal)
            kurtosis = stats.kurtosis(signal)
            
            features.extend([
                heart_rate,
                hr_std,
                mean_val,
                std_val,
                max_val,
                min_val,
                range_val,
                dominant_freq,
                skewness,
                kurtosis,
                np.percentile(signal, 25),
                np.percentile(signal, 75)
            ])
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            features = [0] * 12
        
        return features
    
    def find_peaks_simple(self, signal, threshold):
        peaks = []
        for i in range(1, len(signal)-1):
            if signal[i] > threshold and signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
        return np.array(peaks)
    
    def plot_ecg_signal(self, record, record_name):
        plt.figure(figsize=(12, 8))
        
        n_leads = min(record.n_sig, 6)
        for i in range(n_leads):
            plt.subplot(n_leads, 1, i+1)
            plt.plot(record.p_signal[:, i])
            plt.title(f'Lead {i+1} - {os.path.basename(record_name)}')
            plt.ylabel('Amplitude (mV)')
            plt.grid(True)
        
        plt.xlabel('Samples')
        plt.tight_layout()
        plt.savefig(f'ecg_{os.path.basename(record_name)}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def load_all_records(self, max_records=50):
        all_features = []
        all_labels = []
        all_diagnoses = []
        
        print(f"Loading PTB records from: {self.data_path}")
        print(f"Total records available: {len(self.records)}")
        
        if len(self.records) == 0:
            print("No records found!")
            return np.array([]), np.array([]), []
        
        successful_records = 0
        record_count = min(max_records, len(self.records))
        
        for i, record_full_path in enumerate(tqdm(self.records[:max_records], desc="Processing records")):
            try:
                print(f"\nProcessing record {i+1}/{record_count}: {os.path.basename(record_full_path)}")
                
                header_lines = self.get_patient_info(record_full_path)
                diagnosis = self.extract_diagnosis(header_lines, record_full_path)
                print(f"Diagnosis: {diagnosis}")
                
                record = self.load_record(record_full_path)
                if record is None or record.p_signal is None:
                    continue
                
                lead_to_use = 1 if record.p_signal.shape[1] >= 2 else 0
                signal = record.p_signal[:, lead_to_use]
                
                print(f"Using lead {lead_to_use + 1}, Signal length: {len(signal)} samples")
                
                features = self.extract_features_from_signal(signal, record.fs)
                
                all_features.append(features)
                all_labels.append(diagnosis)
                all_diagnoses.append(diagnosis)
                
                successful_records += 1
                
                if successful_records == 1:
                    self.plot_ecg_signal(record, record_full_path)
                
                print(f"Successfully processed: {os.path.basename(record_full_path)}")
                
            except Exception as e:
                print(f"Error processing record {record_full_path}: {e}")
                continue
        
        print(f"\nSuccessfully processed {successful_records} out of {record_count} records")
        
        if successful_records == 0:
            return np.array([]), np.array([]), []
        
        return np.array(all_features), np.array(all_labels), all_diagnoses

class ECGFeatureModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
    
    def build_model(self):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_data(self, X, y):
        y_encoded = self.le.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y_encoded
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_ptb_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {results[0]:.4f}")
        print(f"Test Accuracy: {results[1]:.4f}")
        
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.le.classes_,
                                  zero_division=0))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return y_pred
    
    def predict_diagnosis(self, features):
        features = self.scaler.transform([features])
        prediction = self.model.predict(features, verbose=0)
        class_idx = np.argmax(prediction)
        diagnosis = self.le.inverse_transform([class_idx])[0]
        confidence = np.max(prediction)
        
        return diagnosis, confidence, prediction[0]

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    data_path = os.path.expanduser('~/Downloads/ptb-diagnostic-ecg-database-1.0.0')
    
    loader = PTBDataLoader(data_path=data_path)
    
    print("Loading PTB database...")
    X, y, diagnoses = loader.load_all_records(max_records=50)
    
    print(f"Successfully loaded {len(X)} records")
    print(f"Unique diagnoses: {np.unique(diagnoses)}")
    print(f"Class distribution:")
    for diagnosis in np.unique(diagnoses):
        count = np.sum(np.array(diagnoses) == diagnosis)
        print(f"  {diagnosis}: {count} samples")
    
    print("\nData loading completed. Use train.py to train the model.")

if __name__ == "__main__":
    main()
