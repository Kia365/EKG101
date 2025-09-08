#!/usr/bin/env python3
"""
Simple EKG Image Predictor - Basic version without matplotlib dependencies

This is a simplified version that focuses on core functionality
without visualization dependencies that might cause compatibility issues.
"""

import numpy as np
import os
import sys

# Try to import tensorflow with error handling
try:
    import tensorflow as tf
    print("TensorFlow imported successfully")
except ImportError as e:
    print(f"TensorFlow import failed: {e}")
    print("Please install TensorFlow: pip install tensorflow")
    sys.exit(1)

# Try to import joblib with error handling
try:
    import joblib
    print("Joblib imported successfully")
except ImportError as e:
    print(f"Joblib import failed: {e}")
    print("Please install joblib: pip install joblib")
    sys.exit(1)

# Import from runtime.py
from runtime import load_model_and_encoder, prep_signal, predict_signal


class SimpleEKGImagePredictor:
    def __init__(self, model_path='ptb_ecg_cnn_model.keras', encoder_path='cnn_label_encoder.pkl'):
        """Initialize the Simple EKG Image Predictor with a trained model.
        
        Args:
            model_path: Path to the .keras model file
            encoder_path: Path to the joblib-saved LabelEncoder
        """
        self.model, self.label_encoder = load_model_and_encoder(model_path, encoder_path)
        self.target_length = 5000  # Expected signal length for the model
        print(f"Model loaded successfully!")
        print(f"Available classes: {self.label_encoder.classes_.tolist()}")
    
    def load_image_simple(self, image_path):
        """Load an EKG image from file using PIL.
        
        Args:
            image_path: Path to the image file (PNG, JPG, etc.)
            
        Returns:
            PIL Image object
        """
        try:
            from PIL import Image
        except ImportError:
            print("PIL (Pillow) not available. Install with: pip install pillow")
            return None
            
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        try:
            image = Image.open(image_path)
            print(f"Successfully loaded image: {image_path}")
            print(f"Image size: {image.size}")
            print(f"Image mode: {image.mode}")
            return image
        except Exception as e:
            raise Exception(f"Error loading image {image_path}: {e}")
    
    def extract_signal_simple(self, image, method='horizontal_projection'):
        """Extract 1D signal from 2D EKG image using simple methods.
        
        Args:
            image: PIL Image object
            method: Method to extract signal ('horizontal_projection', 'center_line')
            
        Returns:
            1D numpy array representing the ECG signal
        """
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Invert if needed (white background to black background)
        if np.mean(img_array) > 128:
            img_array = 255 - img_array
        
        if method == 'horizontal_projection':
            # Sum pixels in each column
            signal = np.sum(img_array, axis=0)
        elif method == 'center_line':
            # Extract center row
            height, width = img_array.shape
            center_row = height // 2
            signal = img_array[center_row, :].astype(np.float32)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
        
        # Normalize
        if np.max(signal) > 0:
            signal = signal / np.max(signal)
        
        return signal.astype(np.float32)
    
    def predict_from_image(self, image_path, extraction_method='horizontal_projection'):
        """Load an EKG image and predict the diagnosis.
        
        Args:
            image_path: Path to the EKG image file
            extraction_method: Method to extract signal from image
            
        Returns:
            dict with prediction results
        """
        # Load image
        image = self.load_image_simple(image_path)
        if image is None:
            return None
        
        # Extract signal
        signal = self.extract_signal_simple(image, extraction_method)
        
        # Make prediction
        predicted_label, confidence, probabilities = predict_signal(
            self.model, self.label_encoder, signal, self.target_length
        )
        
        # Prepare results
        results = {
            'image_path': image_path,
            'extraction_method': extraction_method,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities,
            'class_names': self.label_encoder.classes_.tolist(),
            'extracted_signal': signal
        }
        
        return results
    
    def predict_from_signal(self, signal):
        """Predict diagnosis from a 1D signal array.
        
        Args:
            signal: 1D numpy array representing ECG signal
            
        Returns:
            dict with prediction results
        """
        # Make prediction
        predicted_label, confidence, probabilities = predict_signal(
            self.model, self.label_encoder, signal, self.target_length
        )
        
        # Prepare results
        results = {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities,
            'class_names': self.label_encoder.classes_.tolist(),
            'input_signal': signal
        }
        
        return results
    
    def print_results(self, results, title="PREDICTION RESULTS"):
        """Print prediction results in a formatted way."""
        print("\n" + "="*60)
        print(title)
        print("="*60)
        
        if 'image_path' in results:
            print(f"Image: {results['image_path']}")
            print(f"Extraction Method: {results['extraction_method']}")
        
        print(f"Predicted Diagnosis: {results['predicted_label']}")
        print(f"Confidence: {results['confidence']:.3f}")
        print("\nAll Class Probabilities:")
        for class_name, prob in zip(results['class_names'], results['probabilities']):
            print(f"  {class_name}: {prob:.3f}")
        print("="*60)


def main():
    """Example usage of the Simple EKG Image Predictor."""
    print("Simple EKG Image Predictor - Example Usage")
    print("="*50)
    
    try:
        # Initialize predictor
        predictor = SimpleEKGImagePredictor()
        
        # Example 1: Predict from a synthetic signal
        print("\n1. Testing with synthetic ECG signal...")
        synthetic_signal = np.random.randn(5000).astype(np.float32)
        results = predictor.predict_from_signal(synthetic_signal)
        predictor.print_results(results, "SYNTHETIC SIGNAL PREDICTION")
        
        # Example 2: Test with your EKG image
        print("\n2. Testing with your EKG image...")
        image_path = "/home/kia/Desktop/MI.jpg"
        if os.path.exists(image_path):
            results = predictor.predict_from_image(image_path)
            predictor.print_results(results, 'IMAGE PREDICTION')
        else:
            print(f"Image file not found: {image_path}")
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error in main example: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have the model files:")
        print("   - ptb_ecg_cnn_model.keras")
        print("   - cnn_label_encoder.pkl")
        print("2. Install required packages:")
        print("   pip install tensorflow joblib pillow")
        print("3. Check that your image file exists and is readable")


if __name__ == "__main__":
    main()
