import tensorflow as tf
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os


def focal_loss_mod(gamma=2.0, alpha_vector=None):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)  
        cross_entropy = -y_true * tf.keras.backend.log(y_pred)
        if alpha_vector is not None:
            alpha_tensor = tf.keras.backend.constant(alpha_vector)
            alpha_weight = y_true * alpha_tensor
            alpha_weight = tf.keras.backend.sum(alpha_weight, axis=-1, keepdims=True)
        else:
            alpha_weight = 1.0
        weight = alpha_weight * tf.keras.backend.pow(1 - y_pred, gamma)
        return tf.keras.backend.mean(tf.keras.backend.sum(weight * cross_entropy, axis=-1))
    return loss

aus = sorted([1, 2, 4, 5, 6, 7, 9, 12, 15, 17, 20, 23, 26])

def load_model_with_custom_loss(path):
    print("loading model")
    return load_model(path, custom_objects={'loss': 
                        focal_loss_mod(gamma=2.0, alpha_vector=[0.25, 0.25, 0.7, 0.25, 0.4, 0.25, 0.7])})

def load_data_from_csv(path):
    print(f"\nreading data from: {path}")
    df = pd.read_csv(path, sep=';')
    example = df[[f"AU{i}" for i in aus]].values.astype(np.float32)
    return df, example

def load_manual_samples():
    print("\nusing manual samples")
    sample_1 = {
        1: 5.2, 2: 9.1, 4: 3.4, 5: 8.2, 6: 97.5, 7: 9.9,
        9: 6.7, 12: 100.0, 15: 2.1, 17: 0.8, 20: 5.0, 23: 1.2, 26: 95.0
    }

    sample_2 = {
        1: 0.0, 2: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0,
        9: 0.0, 12: 0.0, 15: 0.0, 17: 0.0, 20: 0.0, 23: 0.0, 26: 0.0
    }

    example = np.array([
        [sample_1[au] for au in aus],
        [sample_2[au] for au in aus]
    ], dtype=np.float32)

    return example

def predict_emotions(model, example, threshold=0.5):
    probs = model.predict(example)
    predicted_indices = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    encoder = LabelEncoder()
    encoder.fit(['Happy', 'Surprise', 'Anger', 'Sad', 'Fear', 'Disgust', 'None'])

    predictions = []
    for i in range(len(example)):
        if confidences[i] >= threshold:
            emotion = encoder.inverse_transform([predicted_indices[i]])[0]
        else:
            emotion = "None"
        predictions.append(emotion)

    return predictions, confidences, probs, encoder

def save_predictions_to_csv(original_df, predictions, confidences, output_csv):
    print(f"\nsaving predictions to: {output_csv}")

    samples = [f"Sample {i+1}" for i in range(len(original_df))]
    df_pred = original_df.copy()
    df_pred.insert(0, "Sample", samples)
    df_pred["Prediction"] = predictions
    df_pred["Confidence"] = [round(c * 100, 2) for c in confidences]
    df_pred.to_csv(output_csv, sep=';', index=False, na_rep="None")

    print("csv saved successfully")

def display_results(predictions, confidences, probs, encoder):
    print("\nmanual prediction results:")

    for i, (emotion, conf) in enumerate(zip(predictions, confidences)):
        print(f"sample {i+1}: {emotion} (confidence: {conf*100:.2f}%)")

    print("\nprobability distribution:")

    for i, prob_vector in enumerate(probs):
        print(f"sample {i+1}:")
        for cls, prob in zip(encoder.classes_, prob_vector):
            print(f"{cls:<10}: {prob*100:.2f}%")
        print()

def main():
    use_csv = False
    threshold = 0.5
    model_path = "training_sessions/train/model/emotion_detection_best.keras"
    csv_path = "dataset_AUs.csv"
    output_csv = f"{os.path.splitext(os.path.basename(csv_path))[0]}_predictions.csv"

    model = load_model_with_custom_loss(model_path)

    if use_csv:
        df, example = load_data_from_csv(csv_path)
    else:
        example = load_manual_samples()

    predictions, confidences, probs, encoder = predict_emotions(model, example, threshold)

    if use_csv:
        save_predictions_to_csv(df, predictions, confidences, output_csv)
    else:
        display_results(predictions, confidences, probs, encoder)

if __name__ == "__main__":
    main()
