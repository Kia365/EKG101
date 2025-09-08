import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import cv2

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


class EKGImagePredictor:
    def __init__(self, model_path='ptb_ecg_cnn_model.keras', encoder_path='cnn_label_encoder.pkl'):
        """Initialize the EKG Image Predictor with a trained model.
        
        Args:
            model_path: Path to the .keras model file
            encoder_path: Path to the joblib-saved LabelEncoder
        """
        self.model, self.label_encoder = load_model_and_encoder(model_path, encoder_path)
        self.target_length = 5000  # Expected signal length for the model
        
    def load_image(self, image_path):
        """Load an EKG image from file.
        
        Args:
            image_path: Path to the image file (PNG, JPG, etc.)
            
        Returns:
            PIL Image object
        """
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
    
    def preprocess_image(self, image):
        """Preprocess image for signal extraction.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image (grayscale, resized)
        """
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Invert if needed (white background to black background)
        if np.mean(img_array) > 128:
            img_array = 255 - img_array
            
        return img_array
    
    def extract_signal_from_image(self, image_array, method='horizontal_projection'):
        """Extract 1D signal from 2D EKG image.
        
        Args:
            image_array: 2D numpy array of the image
            method: Method to extract signal ('horizontal_projection', 'center_line', 'edge_detection')
            
        Returns:
            1D numpy array representing the ECG signal
        """
        if method == 'horizontal_projection':
            return self._extract_by_horizontal_projection(image_array)
        elif method == 'center_line':
            return self._extract_by_center_line(image_array)
        elif method == 'edge_detection':
            return self._extract_by_edge_detection(image_array)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    
    def _extract_by_horizontal_projection(self, image_array):
        """Extract signal using horizontal projection (sum of pixels in each column)."""
        # Sum pixels in each column
        signal = np.sum(image_array, axis=0)
        
        # Normalize
        signal = signal / np.max(signal) if np.max(signal) > 0 else signal
        
        return signal.astype(np.float32)
    
    def _extract_by_center_line(self, image_array):
        """Extract signal by finding the center line of the ECG trace."""
        height, width = image_array.shape
        center_row = height // 2
        
        # Extract center row
        signal = image_array[center_row, :].astype(np.float32)
        
        # Normalize
        signal = signal / np.max(signal) if np.max(signal) > 0 else signal
        
        return signal
    
    def _extract_by_edge_detection(self, image_array):
        """Extract signal using edge detection to find the ECG trace."""
        # Apply edge detection
        edges = cv2.Canny(image_array.astype(np.uint8), 50, 150)
        
        # Find the strongest edge in each column
        signal = np.zeros(edges.shape[1])
        for col in range(edges.shape[1]):
            edge_positions = np.where(edges[:, col] > 0)[0]
            if len(edge_positions) > 0:
                # Use the middle edge position
                signal[col] = edge_positions[len(edge_positions)//2]
        
        # Normalize
        if np.max(signal) > 0:
            signal = signal / np.max(signal)
        
        return signal.astype(np.float32)
    
    def predict_from_image(self, image_path, extraction_method='horizontal_projection', 
                          show_visualization=True):
        """Load an EKG image and predict the diagnosis.
        
        Args:
            image_path: Path to the EKG image file
            extraction_method: Method to extract signal from image
            show_visualization: Whether to show the loaded image and extracted signal
            
        Returns:
            dict with prediction results
        """
        # Load and preprocess image
        image = self.load_image(image_path)
        processed_image = self.preprocess_image(image)
        
        # Extract signal
        signal = self.extract_signal_from_image(processed_image, extraction_method)
        
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
        
        # Show visualization if requested
        if show_visualization:
            self._show_prediction_visualization(image, processed_image, signal, results)
        
        return results
    
    def _show_prediction_visualization(self, original_image, processed_image, signal, results):
        """Show visualization of the image processing and prediction results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original EKG Image')
        axes[0, 0].axis('off')
        
        # Processed image
        axes[0, 1].imshow(processed_image, cmap='gray')
        axes[0, 1].set_title('Processed Image (Grayscale)')
        axes[0, 1].axis('off')
        
        # Extracted signal
        axes[1, 0].plot(signal)
        axes[1, 0].set_title(f'Extracted Signal ({results["extraction_method"]})')
        axes[1, 0].set_xlabel('Sample')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True)
        
        # Prediction results
        axes[1, 1].bar(results['class_names'], results['probabilities'])
        axes[1, 1].set_title('Prediction Probabilities')
        axes[1, 1].set_xlabel('Diagnosis Class')
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Highlight the predicted class
        pred_idx = results['class_names'].index(results['predicted_label'])
        axes[1, 1].bar(pred_idx, results['probabilities'][pred_idx], 
                      color='red', alpha=0.7, label=f'Predicted: {results["predicted_label"]}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print prediction results
        print("\n" + "="*60)
        print("EKG IMAGE PREDICTION RESULTS")
        print("="*60)
        print(f"Image: {results['image_path']}")
        print(f"Extraction Method: {results['extraction_method']}")
        print(f"Predicted Diagnosis: {results['predicted_label']}")
        print(f"Confidence: {results['confidence']:.3f}")
        print("\nAll Class Probabilities:")
        for class_name, prob in zip(results['class_names'], results['probabilities']):
            print(f"  {class_name}: {prob:.3f}")
        print("="*60)
    
    def predict_from_signal(self, signal, show_visualization=True):
        """Predict diagnosis from a 1D signal array.
        
        Args:
            signal: 1D numpy array representing ECG signal
            show_visualization: Whether to show the signal and prediction
            
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
        
        # Show visualization if requested
        if show_visualization:
            self._show_signal_prediction_visualization(signal, results)
        
        return results
    
    def _show_signal_prediction_visualization(self, signal, results):
        """Show visualization of signal prediction results."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Input signal
        axes[0].plot(signal)
        axes[0].set_title('Input ECG Signal')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)
        
        # Prediction results
        axes[1].bar(results['class_names'], results['probabilities'])
        axes[1].set_title('Prediction Probabilities')
        axes[1].set_xlabel('Diagnosis Class')
        axes[1].set_ylabel('Probability')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Highlight the predicted class
        pred_idx = results['class_names'].index(results['predicted_label'])
        axes[1].bar(pred_idx, results['probabilities'][pred_idx], 
                   color='red', alpha=0.7, label=f'Predicted: {results["predicted_label"]}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print prediction results
        print("\n" + "="*60)
        print("ECG SIGNAL PREDICTION RESULTS")
        print("="*60)
        print(f"Predicted Diagnosis: {results['predicted_label']}")
        print(f"Confidence: {results['confidence']:.3f}")
        print("\nAll Class Probabilities:")
        for class_name, prob in zip(results['class_names'], results['probabilities']):
            print(f"  {class_name}: {prob:.3f}")
        print("="*60)


def main():
    """Example usage of the EKG Image Predictor."""
    print("EKG Image Predictor - Example Usage")
    print("="*50)
    
    try:
        # Initialize predictor
        predictor = EKGImagePredictor()
        
        # Example 1: Predict from a synthetic signal
        print("\n1. Testing with synthetic ECG signal...")
        synthetic_signal = np.random.randn(5000).astype(np.float32)
        results = predictor.predict_from_signal(synthetic_signal, show_visualization=True)
        
        # Example 2: If you have an EKG image file, uncomment and modify the path
        # print("\n2. Testing with EKG image...")
        # image_path = "path/to/your/ekg_image.png"  # Replace with actual image path
        # if os.path.exists(image_path):
        #     results = predictor.predict_from_image(image_path, show_visualization=True)
        # else:
        #     print(f"Image file not found: {image_path}")
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error in main example: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have the model files: ptb_ecg_cnn_model.keras and cnn_label_encoder.pkl")
        print("2. Install required packages: pip install tensorflow joblib pillow opencv-python matplotlib")
        print("3. Check that your image file exists and is readable")


if __name__ == "__main__":
    main()
