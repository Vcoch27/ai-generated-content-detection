import numpy as np
import cv2
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops


def get_cv_features(img_256):
    """Trích xuất 27 đặc trưng vật lý (OpenCV)"""
    gray = cv2.cvtColor(img_256, cv2.COLOR_BGR2GRAY)
    features = []
    # Màu sắc (18)
    for color_space in [img_256, cv2.cvtColor(img_256, cv2.COLOR_BGR2HSV)]:
        for i in range(3):
            ch = color_space[:, :, i].flatten()
            features.extend([np.mean(ch), np.std(ch), skew(ch)])
    # Cạnh (2)
    features.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    features.append(np.mean(cv2.Canny(gray, 100, 200)) / 255)
    # Kết cấu GLCM (4)
    glcm = graycomatrix(gray, [5], [0], levels=256, symmetric=True, normed=True)
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        features.append(graycoprops(glcm, prop)[0, 0])
    # Tần số FFT (3)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = 20 * np.log(np.abs(f) + 1)
    features.extend([np.mean(mag), np.std(mag), np.max(mag)])
    return np.array(features).reshape(1, -1)


def get_hybrid_vector(img_path, cnn_model, pca_model, cv_processor, cnn_processor):
    """Kết hợp CV và CNN (đã giảm chiều) thành vector 77 chiều"""
    # 1. Trích xuất CV (27)
    img_cv = cv_processor(img_path)
    cv_feats = get_cv_features(img_cv)

    # 2. Trích xuất CNN (1024)
    img_cnn = cnn_processor(img_path)
    cnn_feats_1024 = cnn_model.predict(img_cnn, verbose=0)

    # 3. Giảm chiều CNN qua PCA (1024 -> 50)
    cnn_feats_50 = pca_model.transform(cnn_feats_1024)

    # 4. Hợp nhất (27 + 50 = 77)
    return np.hstack((cv_feats, cnn_feats_50))