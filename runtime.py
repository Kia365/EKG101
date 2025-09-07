import numpy as np
import tensorflow as tf
import joblib


def load_model_and_encoder(model_path: str = 'ptb_ecg_cnn_model.keras',
                           encoder_path: str = 'cnn_label_encoder.pkl'):
    """Load a saved Keras model and its label encoder.

    Args:
        model_path: Path to the .keras model file.
        encoder_path: Path to the joblib-saved LabelEncoder.

    Returns:
        (model, label_encoder)
    """
    model = tf.keras.models.load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    return model, label_encoder


def prep_signal(signal_1d, target_len: int = 5000) -> np.ndarray:
    """Prepare a 1D ECG signal for inference with the CNN model.

    - Resamples (via index selection) or pads to target_len
    - Normalizes to zero mean and unit variance
    - Adds channel dimension -> shape (1, target_len, 1)

    Args:
        signal_1d: Iterable/array-like of ECG values.
        target_len: Target number of samples expected by the model.

    Returns:
        np.ndarray of shape (1, target_len, 1)
    """
    signal = np.asarray(signal_1d, dtype=np.float32)
    if signal.size > target_len:
        idx = np.linspace(0, signal.size - 1, target_len, dtype=int)
        signal = signal[idx]
    elif signal.size < target_len:
        signal = np.pad(signal, (0, target_len - signal.size))

    signal = (signal - signal.mean()) / (signal.std() + 1e-8)
    return signal.reshape(1, target_len, 1)


def predict_signal(model: tf.keras.Model,
                   label_encoder,
                   signal_1d,
                   target_len: int = 5000):
    """Predict label and confidence for a single 1D ECG signal.

    Args:
        model: Loaded Keras model.
        label_encoder: Loaded LabelEncoder for inverse transform.
        signal_1d: 1D ECG array-like.
        target_len: Target length expected by the model.

    Returns:
        (predicted_label: str, confidence: float, probabilities: np.ndarray)
    """
    x = prep_signal(signal_1d, target_len=target_len)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(probs[pred_idx])
    return pred_label, confidence, probs


def predict_batch(model: tf.keras.Model,
                  label_encoder,
                  signals: np.ndarray,
                  target_len: int = 5000):
    """Predict labels for a batch of 1D ECG signals.

    Args:
        model: Loaded Keras model.
        label_encoder: Loaded LabelEncoder.
        signals: Iterable of 1D signals (list/array) or shape (N, L) array.
        target_len: Target length expected by the model.

    Returns:
        dict with keys: labels (List[str]), confidences (List[float]), probs (np.ndarray)
    """
    signals = list(signals)
    x_batch = np.concatenate([prep_signal(s, target_len=target_len) for s in signals], axis=0)
    probs_batch = model.predict(x_batch, verbose=0)
    idxs = np.argmax(probs_batch, axis=1)
    labels = label_encoder.inverse_transform(idxs)
    confidences = probs_batch[np.arange(len(probs_batch)), idxs].astype(float).tolist()
    return {
        'labels': labels.tolist(),
        'confidences': confidences,
        'probs': probs_batch
    }


if __name__ == '__main__':
    # Example usage (replace with your own paths and signal):
    try:
        model, le = load_model_and_encoder('ptb_ecg_cnn_model.keras', 'cnn_label_encoder.pkl')
        # dummy example signal
        example_signal = np.random.randn(5000).astype(np.float32)
        label, conf, _ = predict_signal(model, le, example_signal)
        print('Predicted:', label, 'Confidence:', conf)
    except Exception as e:
        print('Runtime example failed:', e)


