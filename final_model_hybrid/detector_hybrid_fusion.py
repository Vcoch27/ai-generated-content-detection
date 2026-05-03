import os
import joblib
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from utils.image_processing import preprocess_for_cv, preprocess_for_cnn
from utils.feature_extraction import get_hybrid_vector

# Tắt log thừa của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Đường dẫn models (đảm bảo bro đã để đúng folder models/)
MODEL_DIR = "models"
CNN_PATH = os.path.join(MODEL_DIR, "cnn_feature_extractor_1024.keras")
PCA_PATH = os.path.join(MODEL_DIR, "hybrid_pca_transformer.joblib")
RF_PATH = os.path.join(MODEL_DIR, "ai_detector_final_model_hybrid_fusion.joblib")

print("--- Đang khởi động hệ thống Hybrid Fusion V3 ---")
cnn_extractor = load_model(CNN_PATH)
pca_transformer = joblib.load(PCA_PATH)
rf_classifier = joblib.load(RF_PATH)
print("Hệ thống đã sẵn sàng!\n")


def run_detector():
    root = tk.Tk()
    root.withdraw()

    while True:
        print("=" * 40)
        file_path = filedialog.askopenfilename(title="Chọn ảnh để Hybrid V3 kiểm tra")
        if not file_path: break

        print(f"Đang phân tích: {os.path.basename(file_path)}...")

        try:
            # 1. Trích xuất vector kết tinh 77 chiều
            features_77 = get_hybrid_vector(
                file_path, cnn_extractor, pca_transformer,
                preprocess_for_cv, preprocess_for_cnn
            )

            # 2. Dự đoán bằng Random Forest
            prediction = rf_classifier.predict(features_77)[0]
            confidence = rf_classifier.predict_proba(features_77)[0]

            # 3. Hiển thị kết quả
            label = "NGƯỜI CHỤP (Real)" if prediction == 1 else "AI TẠO (Fake)"
            score = confidence[1] if prediction == 1 else confidence[0]

            print(f"\nKẾT QUẢ: {label}")
            print(f"Độ tin cậy: {score * 100:.2f}%")

        except Exception as e:
            print(f"Lỗi xử lý: {e}")

        cont = input("\nTiếp tục chứ bro? (y/n): ").lower()
        if cont != 'y': break


if __name__ == "__main__":
    run_detector()