# Emotion-Detection-Model

## Introduction
This project implements a facial emotion recognition system based on facial **Action Units (AUs)**, which are numerical indicators of muscle movements in the face according to the Facial Action Coding System (FACS).  
Instead of working directly on raw images, this system takes pre-extracted AU intensity values and predicts one of six basic emotions, using a neural network trained on synthetically generated AU datasets.  

The pipeline consists of three main components:
1. **Synthetic dataset generation** from AU–emotion mapping.
2. **Neural network training** with a custom loss function.
3. **Emotion prediction** on new AU data.

---

## Action Units to Emotion Mapping
Each emotion is defined by a set of **positive AUs** (high activation), **negative AUs** (low activation), and **optional AUs** (can vary).  
The table below shows the mapping used in this project:

| Emotion   | Positive AUs | Negative AUs | Optional AUs |
|-----------|--------------|--------------|--------------|
| **Happy**     | 6, 12       | 20, 26       | 4, 5, 17     |
| **Surprise**  | 5, 26       | 20           | 1, 2         |
| **Angry**     | 4, 23, 7    | 26           | 5, 6, 9, 17  |
| **Sad**       | 4, 15       | 7, 9, 12, 23, 26 | —       |
| **Fear**      | 2, 20, 26   | 9            | 15, 17       |
| **Disgust**   | 6, 9, 17    | 23           | 7, 15        |

*Values are considered "positive" when their AU intensity ≥ 95.0, and "negative" when ≤ 85.0.*

---

## Project Structure

### `generate_data.py`
- Generates a **synthetic dataset** (`dataset_AUs.csv`) containing AU values for each emotion.
- Ensures a balanced dataset (default: 2000 samples per emotion).
- Uses random AU intensities within defined ranges for positive, negative, and optional units.
- Prevents overlapping definitions where a sample could satisfy a higher-priority emotion's conditions.

### `train.py`
- Loads the dataset (`dataset_AUs.csv`).
- Encodes labels and splits into **train**, **validation**, and **test** sets.
- Defines and trains a **fully connected neural network** with:
  - Dense + LeakyReLU + Batch Normalization layers
  - Dropout for regularization
  - **Focal Loss** to handle class imbalance
- Saves:
  - Best and last trained models
  - Accuracy/loss curves
  - Confusion matrices (absolute and normalized)
  - Classification reports & metrics (Precision, Recall, F1, MCC, Kappa, etc.)
  - Confidence threshold analysis plots

### `predict.py`
- Loads the trained model with custom loss function.
- Allows prediction from:
  - A CSV file containing AU values.
  - Manually defined AU samples.
- Outputs:
  - Predicted emotion for each sample.
  - Confidence score.
  - Full probability distribution.
- Optionally saves predictions to a new CSV file.

---

## Requirements

Install all dependencies:

```
pip install -r requirements.txt
```

---

## Usage

### 1. Generate Dataset
```
python generate_data.py
```

This will create **dataset_AUs.csv**.

### 2. Train Model
```
python train.py
```

Training results will be saved under **training_sessions/**.

### 3. Predict Emotions
```
python predict.py
```

You can toggle between CSV input and manual samples by changing the **use_csv** flag in the script.

---

## Output Example

Training session folder structure:

```
training_sessions/
  ├── train/
  │   ├── model/
  │   │   ├── emotion_detection_best.keras
  │   │   └── emotion_detection_last.keras
  │   ├── accuracy_curve_train.png
  │   ├── accuracy_curve_val.png
  │   ├── confusion_matrix.png
  │   ├── confusion_matrix_normalized.png
  │   ├── classification_report.txt
  │   ├── metrics_barplot.png
  │   └── extra_metrics.txt
```