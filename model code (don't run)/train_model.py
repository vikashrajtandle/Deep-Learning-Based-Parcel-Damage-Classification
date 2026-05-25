import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt

# ==============================
# GLOBAL PATH CONFIGURATION
# ==============================

DATASET_PATH = r"media\damaged-and-intact-packages"
MODEL_DIR = r"media\trainedmodel"

IMG_HEIGHT = 224
IMG_WIDTH = 224
CNN_BATCH_SIZE = 16
CNN_EPOCHS = 20

SVM_IMG_SIZE = 128

PURE_CNN_MODEL_PATH = os.path.join(MODEL_DIR, "pure_cnn_efficientnet_model.h5")
PURE_SVM_MODEL_PATH = os.path.join(MODEL_DIR, "pure_svm_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

print("\n====================================================")
print(" PARCEL DAMAGE CLASSIFICATION - FULL TRAINING ")
print(" PURE CNN  +  PURE SVM ")
print("====================================================\n")

# =====================================================
# PART 1 — PURE CNN MODEL (EfficientNetB3)
# =====================================================

print("\n================ PURE CNN TRAINING =================\n")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=CNN_BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=CNN_BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print("\n[INFO] Building High-End CNN Model (EfficientNetB3)...")

base_model = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

cnn_model = Model(inputs=base_model.input, outputs=output)

cnn_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()

checkpoint = ModelCheckpoint(PURE_CNN_MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\n[INFO] Training Pure CNN Model...\n")

history = cnn_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=CNN_EPOCHS,
    callbacks=[checkpoint, earlystop]
)

cnn_model.save(PURE_CNN_MODEL_PATH)
print(f"\n[SUCCESS] Pure CNN Model Saved → {PURE_CNN_MODEL_PATH}")

# Plot CNN metrics
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Pure CNN Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Pure CNN Loss')
plt.legend()

plt.show()

# =====================================================
# PART 2 — PURE SVM MODEL (Classical ML)
# =====================================================

print("\n================ PURE SVM TRAINING =================\n")

def load_images(folder):
    data = []
    labels = []
    classes = sorted(os.listdir(folder))

    print(f"[INFO] Classes Found: {classes}")

    for idx, cls in enumerate(classes):
        cls_path = os.path.join(folder, cls)
        print(f"[INFO] Loading: {cls_path}")
        
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (SVM_IMG_SIZE, SVM_IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = img.flatten()

                data.append(img)
                labels.append(idx)
            except:
                continue

    return np.array(data), np.array(labels)

print("\n[INFO] Loading Dataset for SVM...")

X, y = load_images(DATASET_PATH)

print(f"[INFO] Total Samples: {X.shape[0]}")
print(f"[INFO] Feature Vector Size: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n[INFO] Training Pure SVM Classifier...")

svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True
)

svm_model.fit(X_train, y_train)

print("\n[SUCCESS] SVM Training Completed!")

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n[SUCCESS] Pure SVM Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

joblib.dump(svm_model, PURE_SVM_MODEL_PATH)

print(f"\n[SUCCESS] Pure SVM Model Saved → {PURE_SVM_MODEL_PATH}")

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n====================================================")
print(" FULL TRAINING COMPLETED SUCCESSFULLY ")
print("====================================================\n")

print("Saved Models:")
print(f"✔ Pure CNN → {PURE_CNN_MODEL_PATH}")
print(f"✔ Pure SVM → {PURE_SVM_MODEL_PATH}")
