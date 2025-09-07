# ECG CNN Model for Google Colab

This project contains a Convolutional Neural Network (CNN) implementation for ECG signal classification, optimized for training on Google Colab with GPU support.

## Files Overview

### Core Files
- **`EKG_CNN.py`** - Main CNN implementation with data loading and model classes
- **`train_cnn_colab.py`** - Training script optimized for Google Colab
- **`ECG_CNN_Colab.ipynb`** - Jupyter notebook for interactive training in Colab

### Original Files (for reference)
- **`EKG.py`** - Original feature-based model implementation
- **`train.py`** - Original training script

## Key Features

### CNN Architecture
- **1D Convolutional layers** optimized for ECG signals
- **Batch normalization** and **dropout** for regularization
- **Global average pooling** to reduce overfitting
- **Mixed precision training** for GPU acceleration

### Data Processing
- **Signal preprocessing** with baseline drift removal
- **Fixed-length signals** (5000 samples) for consistent CNN input
- **Multi-lead support** with automatic lead selection
- **Synthetic data generation** for testing when real data is unavailable

### GPU Optimization
- **Automatic GPU detection** and configuration
- **Memory growth** to prevent OOM errors
- **Mixed precision** (float16) for faster training
- **Optimized batch sizes** for GPU memory

## How to Use

### Option 1: Google Colab Notebook (Recommended)
1. Upload `ECG_CNN_Colab.ipynb` to Google Colab
2. Enable GPU runtime (Runtime → Change runtime type → GPU)
3. Run all cells sequentially
4. The notebook will automatically:
   - Install required packages
   - Configure GPU settings
   - Load or create synthetic data
   - Train the CNN model
   - Evaluate and save results

### Option 2: Python Script
1. Upload `EKG_CNN.py` and `train_cnn_colab.py` to Colab
2. Run: `python train_cnn_colab.py`

### Option 3: Local Training
1. Install dependencies: `pip install tensorflow wfdb scikit-learn tqdm`
2. Download and extract PTB dataset to `~/Downloads/ptb-diagnostic-ecg-database-1.0.0/`
3. Run: `python train_cnn_colab.py`

## Model Architecture

```
Input: (5000, 1) - ECG signal
├── Conv1D(64, kernel=15) + BatchNorm + MaxPool + Dropout
├── Conv1D(128, kernel=10) + BatchNorm + MaxPool + Dropout  
├── Conv1D(256, kernel=5) + BatchNorm + MaxPool + Dropout
├── Conv1D(512, kernel=3) + BatchNorm + GlobalAvgPool + Dropout
├── Dense(256) + BatchNorm + Dropout
├── Dense(128) + BatchNorm + Dropout
└── Dense(num_classes) - Softmax output
```

## Data Requirements

### PTB Dataset (Preferred)
- Download from: https://physionet.org/content/ptb-db/1.0.0/
- Extract to `~/Downloads/ptb-diagnostic-ecg-database-1.0.0/` on your local machine
- For Colab: Upload to `/content/ptb-diagnostic-ecg-database-1.0.0/`
- Contains real ECG recordings with clinical diagnoses

### Synthetic Data (Fallback)
- Automatically generated if PTB dataset not found
- Creates realistic ECG patterns for different conditions
- Useful for testing and demonstration

## Performance Features

### Training Optimizations
- **Early stopping** to prevent overfitting
- **Learning rate reduction** on plateau
- **Model checkpointing** to save best weights
- **Stratified train/test split** for balanced evaluation

### Evaluation Metrics
- **Accuracy, Precision, Recall, F1-score**
- **Confusion matrix** visualization
- **Classification report** with per-class metrics
- **Sample predictions** with confidence scores

## Output Files

After training, the following files are saved:
- `ptb_ecg_cnn_model.keras` - Trained CNN model
- `cnn_label_encoder.pkl` - Label encoder for predictions
- `cnn_training_history.png` - Training curves plot
- `best_ptb_cnn_model.keras` - Best model checkpoint

## Usage Examples

### Load and Use Trained Model
```python
import tensorflow as tf
import joblib
import numpy as np

# Load model and encoder
model = tf.keras.models.load_model('ptb_ecg_cnn_model.keras')
le = joblib.load('cnn_label_encoder.pkl')

# Predict on new signal
def predict_ecg(signal):
    # Preprocess signal (normalize, resize to 5000 samples)
    signal = signal.reshape(1, 5000, 1)
    prediction = model.predict(signal)
    class_idx = np.argmax(prediction)
    diagnosis = le.inverse_transform([class_idx])[0]
    confidence = np.max(prediction)
    return diagnosis, confidence
```

## Troubleshooting

### Common Issues
1. **GPU not detected**: Ensure GPU runtime is enabled in Colab
2. **Out of memory**: Reduce batch size or use CPU
3. **No data found**: Check PTB dataset path or use synthetic data
4. **Import errors**: Install missing packages with pip

### Performance Tips
- Use GPU runtime for faster training
- Increase batch size if you have more GPU memory
- Adjust signal length based on your data
- Use data augmentation for small datasets

## Comparison with Original Model

| Feature | Original (EKG.py) | CNN (EKG_CNN.py) |
|---------|------------------|------------------|
| Input | Extracted features (12) | Raw signals (5000) |
| Architecture | Dense layers | 1D CNN + Dense |
| Training | CPU optimized | GPU optimized |
| Performance | Good for small datasets | Better for large datasets |
| Interpretability | Feature-based | Signal-based |

## Next Steps

1. **Data Augmentation**: Add noise, time stretching, etc.
2. **Multi-lead Fusion**: Use all ECG leads simultaneously  
3. **Attention Mechanisms**: Add attention layers for better focus
4. **Transfer Learning**: Pre-train on larger ECG datasets
5. **Real-time Inference**: Optimize for mobile/edge deployment
