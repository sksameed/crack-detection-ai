import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input

# ==============================
# LOAD MODEL
# ==============================
model = tf.keras.models.load_model(
    "crack_pretrained_model.keras",
    compile=False
)

IMG_SIZE = 256

# ==============================
# PREDICT FUNCTION
# ==============================
def predict_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for model
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    # 🔥 IMPORTANT: same preprocessing as training
    img_processed = preprocess_input(img_resized.astype(np.float32))

    img_input = np.expand_dims(img_processed, axis=0)

    pred = model.predict(img_input)[0]

    # 🔥 Use same threshold as training
    mask = (pred > 0.3).astype(np.uint8)

    # Resize mask back to original size
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    return img_rgb, mask


# ==============================
# SEVERITY FUNCTION
# ==============================
def get_severity(mask):
    crack_percentage = mask.mean() * 100

    if crack_percentage < 1:
        severity = "LOW"
    elif crack_percentage < 5:
        severity = "MEDIUM"
    else:
        severity = "HIGH"

    return crack_percentage, severity


# ==============================
# SHOW RESULT
# ==============================
def show_result(image_path):
    img, mask = predict_image(image_path)

    crack_percentage, severity = get_severity(mask)

    # 🔥 Create overlay (RED cracks)
    overlay = img.copy()
    overlay[mask > 0.5] = [255, 0, 0]

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title(f"Overlay ({severity})")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()

    print(f"Crack Percentage: {crack_percentage:.2f}%")
    print(f"Risk Severity   : {severity}")


# ==============================
# RUN
# ==============================
show_result("test.jpeg")