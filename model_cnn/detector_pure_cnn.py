import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- 1. CẤU HÌNH THÔNG SỐ ---
MODEL_NAME = 'ai_detector_model_pure_cnn.keras'
IMG_SIZE = (224, 224)  # Kích thước chuẩn của MobileNetV2


def main():
    # Kiểm tra sự tồn tại của file models
    if not os.path.exists(MODEL_NAME):
        print(f"Lỗi: Không tìm thấy file {MODEL_NAME} trong thư mục này!")
        print("Bro nhớ tải file từ Drive về và để cùng thư mục với file code này nhé.")
        return

    print(f"--- Đang tải mô hình: {MODEL_NAME} ---")
    try:
        model = load_model(MODEL_NAME)
        print("Mô hình CNN đã sẵn sàng!")
    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        return

    # Tạo cửa sổ ẩn để dùng hộp thoại chọn file
    root = tk.Tk()
    root.withdraw()

    while True:
        print("\n" + "=" * 40)
        print("Mời bro chọn một tấm ảnh để CNN kiểm tra...")

        # Mở hộp thoại chọn file
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh cần kiểm tra (CNN Detector)",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )

        if not file_path:
            print("Chưa chọn ảnh. Kết thúc chương trình.")
            break

        print(f"Đang phân tích: {os.path.basename(file_path)}")

        try:
            # --- 2. TIỀN XỬ LÝ ẢNH (QUAN TRỌNG: PHẢI GIỐNG LÚC TRAIN) ---
            # Load ảnh và resize về 224x224
            img = image.load_img(file_path, target_size=IMG_SIZE)

            # Chuyển ảnh thành mảng numpy
            img_array = image.img_to_array(img)

            # Thêm chiều batch (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)

            # Chuẩn hóa pixel về [0, 1]
            img_array /= 255.0

            # --- 3. DỰ ĐOÁN ---
            prediction = model.predict(img_array, verbose=0)[0][0]

            print("-" * 30)
            # Vì lúc train chúng ta dùng class_mode='binary'
            # Thông thường: Fake (AI) = 0, Real (Thật) = 1 (tùy thuộc vào thứ tự folder alphabet)
            # Theo logic ImageDataGenerator: 'fake' đứng trước 'real' -> fake=0, real=1

            if prediction > 0.5:
                confidence = prediction * 100
                print(f"KẾT QUẢ: >>> ẢNH CHỤP THẬT (REAL) <<<")
                print(f"Độ tin cậy: {confidence:.2f}%")
            else:
                confidence = (1 - prediction) * 100
                print(f"KẾT QUẢ: >>> ẢNH DO AI TẠO RA (FAKE) <<<")
                print(f"Độ tin cậy: {confidence:.2f}%")
            print("-" * 30)

        except Exception as e:
            print(f"Có lỗi xảy ra khi xử lý ảnh: {e}")

        ans = input("Tiếp tục kiểm tra ảnh khác chứ bro? (y/n): ")
        if ans.lower() != 'y':
            break

    print("Cảm ơn bro đã tin dùng mô hình pure_cnn!")


if __name__ == "__main__":
    main()