import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def preprocess_for_cv(img_path):
    """Tiền xử lý ảnh cho nhánh OpenCV"""
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, (256, 256))
    return img


def preprocess_for_cnn(img_path):
    """Tiền xử lý ảnh cho nhánh CNN (MobileNetV2)"""
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array