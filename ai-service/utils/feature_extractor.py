import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew

# Định nghĩa danh sách tên cột (PHẢI GIỐNG 100% KHI TRAIN)
FEATURE_NAMES = [
    'R_mean', 'R_std', 'R_skew', 'G_mean', 'G_std', 'G_skew', 'B_mean', 'B_std', 'B_skew',
    'H_mean', 'H_std', 'H_skew', 'S_mean', 'S_std', 'S_skew', 'V_mean', 'V_std', 'V_skew',
    'Sharpness', 'Edge_Density', 'GLCM_Contrast', 'GLCM_Corr', 'GLCM_Energy',
    'GLCM_Homog', 'FFT_Mean', 'FFT_Std', 'FFT_Max'
]


def extract_features(img):
    """
    Trích xuất đặc trưng từ ảnh.
    Input: ảnh BGR từ OpenCV (numpy array)
    Output: danh sách 27 đặc trưng hoặc None nếu lỗi
    """
    try:
        # Tiền xử lý
        img = cv2.resize(img, (256, 256))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = []

        # Nhóm Màu sắc (18 features)
        for color_space in [img, cv2.cvtColor(img, cv2.COLOR_BGR2HSV)]:
            for i in range(3):
                ch = color_space[:, :, i].flatten()
                features.extend([np.mean(ch), np.std(ch), skew(ch)])

        # Nhóm Độ sắc nét & Cạnh (2 features)
        features.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = cv2.Canny(gray, 100, 200)
        features.append(np.mean(edges) / 255)

        # Nhóm Kết cấu GLCM (4 features)
        glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
            features.append(graycoprops(glcm, prop)[0, 0])

        # Nhóm Tần số FFT (3 features)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        features.extend([np.mean(magnitude_spectrum), np.std(magnitude_spectrum), np.max(magnitude_spectrum)])

        return features
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh: {e}")
        return None


def create_feature_dataframe(features):
    """
    Chuyển danh sách features thành DataFrame với đúng tên cột.
    """
    if features is None or len(features) != len(FEATURE_NAMES):
        return None
    return pd.DataFrame([features], columns=FEATURE_NAMES)
