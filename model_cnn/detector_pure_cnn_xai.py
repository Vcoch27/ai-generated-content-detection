import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- 1. CẤU HÌNH THÔNG SỐ ---
MODEL_NAME = 'ai_detector_model_pure_cnn.keras'
IMG_SIZE = (224, 224)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Tạo một model phụ lấy đầu vào là ảnh và đầu ra là lớp tích chập cuối cùng + đầu ra dự đoán
    # Đối với MobileNetV2, chúng ta cần truy cập vào layer mobilenetv2 bên trong nếu nó bị đóng gói

    # Tìm lớp mobilenetv2 trong model của bro
    base_model = None
    for layer in model.layers:
        if 'mobilenetv2' in layer.name:
            base_model = layer
            break

    if base_model:
        # Nếu model được build theo kiểu Functional API hoặc có chứa base_model
        grad_model = tf.keras.models.Model(
            [base_model.input],
            [base_model.get_layer(last_conv_layer_name).output, base_model.output]
        )
    else:
        # Nếu là model phẳng (Sequential)
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

    # 2. Tính toán Gradient với logic đảo ngược cho AI
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)

        # Lấy giá trị dự đoán (vì là Sigmoid nên chỉ có 1 node duy nhất ở preds[0][0])
        score = preds[0][0]

        # LOGIC QUAN TRỌNG:
        # Nếu score < 0.5 (AI): Ta muốn tìm đặc trưng làm GIẢM score (tăng tính AI)
        # Nếu score > 0.5 (REAL): Ta muốn tìm đặc trưng làm TĂNG score (tăng tính REAL)
        if score < 0.5:
            target_score = 1 - score  # Đảo ngược để đạo hàm của vùng AI trở thành dương
        else:
            target_score = score

    # 3. Tính đạo hàm của target_score đối với lớp tích chập cuối
    grads = tape.gradient(target_score, last_conv_layer_output)

    # 4. Trọng số alpha (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Nhân trọng số và cộng dồn các feature maps
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. ReLU và Chuẩn hóa
    # ReLU giữ lại những vùng ĐÓNG GÓP DƯƠNG vào việc ra quyết định (AI hoặc REAL)
    heatmap = tf.maximum(heatmap, 0)

    # Tránh chia cho 0 nếu heatmap trắng trơn
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        max_val = 1e-10

    heatmap = heatmap / max_val
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load ảnh gốc bằng OpenCV
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # Resize heatmap về kích thước ảnh gốc (224x224)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Chuyển đổi heatmap sang hệ màu RGB (dạng Heatmap đỏ xanh)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Đè Heatmap lên ảnh gốc (Overlay)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Hiển thị bằng Matplotlib cho đẹp
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Bản đồ nhiệt (Heatmap)")
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Giải thích vùng AI (Overlay)")
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    if not os.path.exists(MODEL_NAME):
        print(f"Lỗi: Không tìm thấy file {MODEL_NAME}!")
        return

    print(f"--- Đang tải mô hình: {MODEL_NAME} ---")
    model = load_model(MODEL_NAME)

    # Xác định lớp tích chập cuối cùng để soi
    # Với MobileNetV2 chuẩn, lớp này thường là 'out_relu'
    last_conv_layer_name = "out_relu"

    root = tk.Tk()
    root.withdraw()

    while True:
        print("\n" + "=" * 40)
        file_path = filedialog.askopenfilename(title="Chọn ảnh kiểm tra XAI")

        if not file_path: break

        try:
            # Tiền xử lý
            img = image.load_img(file_path, target_size=IMG_SIZE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Dự đoán
            preds = model.predict(img_array, verbose=0)
            prediction = preds[0][0]

            print("-" * 30)
            if prediction > 0.5:
                print(f"KẾT QUẢ: >>> ẢNH CHỤP THẬT (REAL) <<< ({prediction * 100:.2f}%)")
            else:
                print(f"KẾT QUẢ: >>> ẢNH DO AI TẠO RA (FAKE) <<< ({(1 - prediction) * 100:.2f}%)")

            # --- XỬ LÝ XAI (GRAD-CAM) ---
            print("Đang tạo bản đồ nhiệt giải thích...")
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            save_and_display_gradcam(file_path, heatmap)

        except Exception as e:
            print(f"Lỗi xử lý: {e}")

        if input("Tiếp tục? (y/n): ").lower() != 'y': break


if __name__ == "__main__":
    main()