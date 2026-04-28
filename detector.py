import cv2
import numpy as np
import joblib
import tkinter as tk
from tkinter import filedialog
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew
import os


# --- 1. KHAI BÁO HÀM TRÍCH XUẤT ĐẶC TRƯNG (PHẢI GIỐNG 100% KHI TRAIN) ---
def extract_features(img):
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

import pandas as pd

# Định nghĩa lại danh sách tên cột y hệt lúc Train (phải đúng thứ tự)
feature_names = [
    'R_mean', 'R_std', 'R_skew', 'G_mean', 'G_std', 'G_skew', 'B_mean', 'B_std', 'B_skew',
    'H_mean', 'H_std', 'H_skew', 'S_mean', 'S_std', 'S_skew', 'V_mean', 'V_std', 'V_skew',
    'Sharpness', 'Edge_Density', 'GLCM_Contrast', 'GLCM_Corr', 'GLCM_Energy',
    'GLCM_Homog', 'FFT_Mean', 'FFT_Std', 'FFT_Max'
]

# --- 2. CHƯƠNG TRÌNH CHÍNH ---
def main():
    # Đường dẫn file model
    model_path = 'ai_detector_model_v2.joblib'

    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file {model_path} trong thư mục này!")
        return

    print("Đang tải mô hình AI...")
    model = joblib.load(model_path)
    print("Mô hình đã sẵn sàng!")

    # Tạo cửa sổ ẩn để dùng hộp thoại chọn file
    root = tk.Tk()
    root.withdraw()

    while True:
        print("\n" + "=" * 30)
        print("Mời bạn chọn một tấm ảnh để kiểm tra...")

        # Mở hộp thoại chọn file
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh cần kiểm tra",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not file_path:
            print("Bạn chưa chọn ảnh. Kết thúc chương trình.")
            break

        print(f"Đang phân tích: {os.path.basename(file_path)}")

        # Đọc ảnh
        img = cv2.imread(file_path)
        if img is None:
            print("Không thể đọc được file ảnh này!")
            continue

        # Trích xuất đặc trưng
        features = extract_features(img)

        if features:
            # Tạo một DataFrame 1 dòng từ list features và gán tên cột
            features_df = pd.DataFrame([features], columns=feature_names)

            # Đưa DataFrame này vào dự đoán
            prediction = model.predict(features_df)[0]
            probs = model.predict_proba(features_df)[0]

            print("-" * 30)
            if prediction == 1:
                confidence = probs[1] * 100
                print(f"KẾT QUẢ: >>> ẢNH DO AI TẠO RA <<<")
                print(f"Độ tin cậy: {confidence:.2f}%")
            else:
                confidence = probs[0] * 100
                print(f"KẾT QUẢ: >>> ẢNH CHỤP THẬT (REAL) <<<")
                print(f"Độ tin cậy: {confidence:.2f}%")
            print("-" * 30)

            # Hiển thị ảnh cho người dùng xem
            # cv2.imshow("Anh dang kiem tra", cv2.resize(img, (500, 500)))
            # cv2.waitKey(0) # Nhấn phím bất kỳ để đóng ảnh và tiếp tục
            # cv2.destroyAllWindows()

        ans = input("Bạn có muốn kiểm tra ảnh khác không? (y/n): ")
        if ans.lower() != 'y':
            break

    print("Cảm ơn bro đã sử dụng chương trình!")


if __name__ == "__main__":
    main()