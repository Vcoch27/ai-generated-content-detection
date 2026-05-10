import cv2
import numpy as np
import base64
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


def encode_image_to_base64(image_np):
    """Chuyển mảng numpy ảnh sang chuỗi Base64 để gửi qua API"""
    _, buffer = cv2.imencode('.png', image_np)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"