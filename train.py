import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, LeakyReLU, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score,\
                            matthews_corrcoef, cohen_kappa_score, hamming_loss, log_loss, precision_recall_curve


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

def contar_amostras_por_classe(y_data, label_encoder):
    labels = np.argmax(y_data, axis=1)
    counts = pd.Series(labels).value_counts().sort_index()
    return {
        label_encoder.classes_[i]: counts.get(i, 0)
        for i in range(len(label_encoder.classes_))
    }

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", buffering=1, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


base_dir = 'training_sessions'
os.makedirs(base_dir, exist_ok=True)

i = 1
while True:
    train_dir = os.path.join(base_dir, f"train{i if i > 1 else ''}")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        break
    i += 1

print(f"\n\nsaving all outputs to: {train_dir}\n")

log_path = os.path.join(train_dir, "terminal_output.txt")
sys.stdout = TeeLogger(log_path)
sys.stderr = sys.stdout

# carregar dataset
df = pd.read_csv('dataset_AUs.csv', sep=';', dtype={'Emotion': str})
df['Emotion'] = df['Emotion']
X = df.drop('Emotion', axis=1).values.astype(np.float32)
y = df['Emotion'].astype(str)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"dataset loaded with {df.shape[0]} samples and {df.shape[1] - 1} AUs")
print(f"\nclasses found: {list(label_encoder.classes_)}")

# 80% treino_val + 20% teste
X_train_val, X_test, y_train_val, y_test = train_test_split( X, y_categorical, test_size=0.2, random_state=42, stratify=np.argmax(y_categorical, axis=1))

# dentro dos 80%, separar 10% para validação → 72% treino / 8% validação
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42, stratify=np.argmax(y_train_val, axis=1))

print("\ndataset split into training, validation, and test sets:")
print(f"   • training:   {X_train.shape[0]} samples")
print(f"   • validation: {X_val.shape[0]} samples")
print(f"   • test:       {X_test.shape[0]} samples\n")

# rede neuronal
model = Sequential([
    Input(shape=(X.shape[1],)),

    Dense(256, kernel_regularizer=l2(0.001)),
    LeakyReLU(negative_slope=0.01),
    BatchNormalization(),
    Dropout(0.5),

    Dense(128, kernel_regularizer=l2(0.001)),
    LeakyReLU(negative_slope=0.01),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, kernel_regularizer=l2(0.001)),
    LeakyReLU(negative_slope=0.01),

    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss=focal_loss_mod(gamma=2.0),
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model_dir = os.path.join(train_dir, 'model')
os.makedirs(model_dir, exist_ok=True)

last_model_path = os.path.join(model_dir, 'emotion_detection_last.keras')
best_model_path = os.path.join(model_dir, 'emotion_detection_best.keras')

checkpoint_callback = ModelCheckpoint(
    filepath=best_model_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

print("\nstarting model training\n")

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint_callback, early_stopping]
)

print("\nmodel training completed")

contagem_treino = contar_amostras_por_classe(y_train, label_encoder)
contagem_validacao = contar_amostras_por_classe(y_val, label_encoder)
contagem_teste = contar_amostras_por_classe(y_test, label_encoder)

with open(os.path.join(train_dir, 'samples_per_class.txt'), 'w', encoding='utf-8') as f:
    f.write("number of samples per class:\n\n")
    
    f.write("=== training ===\n")
    for k, v in contagem_treino.items():
        f.write(f"{k}: {v}\n")
    
    f.write("\n=== validation ===\n")
    for k, v in contagem_validacao.items():
        f.write(f"{k}: {v}\n")
    
    f.write("\n=== test ===\n")
    for k, v in contagem_teste.items():
        f.write(f"{k}: {v}\n")

# accuracy (train)
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig(os.path.join(train_dir, 'accuracy_curve_train.png'))
plt.close()

# accuracy (validation)
plt.figure()
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig(os.path.join(train_dir, 'accuracy_curve_val.png'))
plt.close()

# loss (train)
plt.figure()
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(os.path.join(train_dir, 'loss_curve_train.png'))
plt.close()

# loss (validation)
plt.figure()
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(os.path.join(train_dir, 'loss_curve_val.png'))
plt.close()

print("\ntraining and validation curves saved separately")

model.save(last_model_path)
print("\nmodel saved\n")

# matriz de confusão
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\ngenerating confusion matrix")

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Absolute)")
plt.savefig(os.path.join(train_dir, 'confusion_matrix.png'))
plt.close()

# Matriz de confusão normalizada
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=label_encoder.classes_)
disp_norm.plot(cmap='Blues', values_format='.2f')
plt.title("Confusion Matrix (Normalized)")
plt.savefig(os.path.join(train_dir, 'confusion_matrix_normalized.png'))
plt.close()

print("\nconfusion matrix saved")

# gerar o relatório
report = classification_report(
    y_true_classes, y_pred_classes,
    target_names=label_encoder.classes_,
    output_dict=True
)

# salvar o relatório como texto
report_str = classification_report(
    y_true_classes, y_pred_classes,
    target_names=label_encoder.classes_
)

with open(os.path.join(train_dir, 'classification_report.txt'), 'w') as f:
    f.write(report_str)

print("\nclassification report saved")

# criar gráfico de barras com precision, recall e f1-score por classe
metrics = ['precision', 'recall', 'f1-score']
classes = label_encoder.classes_

x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    values = [report[cls][metric] for cls in classes]
    plt.bar(x + i * width, values, width=width, label=metric.capitalize())

plt.xticks(x + width, classes)
plt.ylim(0, 1.05)
plt.title('Classification Metrics by Emotion')
plt.xlabel('Emotion')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(train_dir, 'metrics_barplot.png'))
plt.close()

print("\nmetrics bar plot saved")

# salvar métricas adicionais
mcc = matthews_corrcoef(y_true_classes, y_pred_classes)
kappa = cohen_kappa_score(y_true_classes, y_pred_classes)
h_loss = hamming_loss(y_true_classes, y_pred_classes)
logloss = log_loss(y_test, y_pred_probs)

with open(os.path.join(train_dir, 'extra_metrics.txt'), 'w') as f:
    f.write(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}\n")
    f.write(f"Cohen's Kappa: {kappa:.4f}\n")
    f.write(f"Hamming Loss: {h_loss:.4f}\n")
    f.write(f"Log Loss: {logloss:.4f}\n")

print("\nextra metrics saved")

# guardar os dados originais do teste para uso nas curvas
y_test_original = y_test.copy()
y_pred_probs_original = y_pred_probs.copy()

# curvas precision, recall e f1 vs confiança
classes = label_encoder.classes_

# precision vs confidence
plt.figure(figsize=(8, 5))
for i, cls in enumerate(classes):
    precision, recall, thresholds = precision_recall_curve(y_test_original[:, i], y_pred_probs_original[:, i])
    plt.plot(thresholds, precision[:-1], label=cls)
plt.title("Precision vs Confidence Threshold")
plt.xlabel("Confidence Threshold")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(train_dir, 'precision_vs_confidence.png'))
plt.close()

# recall vs confidence
plt.figure(figsize=(8, 5))
for i, cls in enumerate(classes):
    precision, recall, thresholds = precision_recall_curve(y_test_original[:, i], y_pred_probs_original[:, i])
    plt.plot(thresholds, recall[:-1], label=cls)
plt.title("Recall vs Confidence Threshold")
plt.xlabel("Confidence Threshold")
plt.ylabel("Recall")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(train_dir, 'recall_vs_confidence.png'))
plt.close()

# f1-score vs confidence
plt.figure(figsize=(8, 5))
for i, cls in enumerate(classes):
    precision, recall, thresholds = precision_recall_curve(y_test_original[:, i], y_pred_probs_original[:, i])
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)
    plt.plot(thresholds, f1[:-1], label=cls)
plt.title("F1-Score vs Confidence Threshold")
plt.xlabel("Confidence Threshold")
plt.ylabel("F1-Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(train_dir, 'f1_vs_confidence.png'))
plt.close()

# precision–recall curves
plt.figure(figsize=(8, 6))
for i, cls in enumerate(label_encoder.classes_):
    precision, recall, _ = precision_recall_curve(y_test_original[:, i], y_pred_probs_original[:, i])
    plt.plot(recall, precision, label=cls)

plt.title("Precision–Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(train_dir, 'precision_recall_curve.png'))
plt.close()

print("\nconfidence threshold curves saved")
print(f"\nall outputs saved at: {train_dir}\n")
