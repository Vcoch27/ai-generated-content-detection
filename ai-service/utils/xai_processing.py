import tensorflow as tf
import cv2
import numpy as np

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    """Tạo heatmap Grad-CAM dựa trên logic AI/REAL """
    # 1. Tạo model gradient
    target_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [target_layer.output, model.output]
    )

    # 2. Tính Gradient
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        score = preds[0][0]
        # Logic: Nếu AI (score < 0.5), tìm cái làm tăng 'tính AI' (1-score)
        target = 1 - score if score < 0.5 else score

    grads = tape.gradient(target, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 3. Tính toán Heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 4. ReLU và Chuẩn hóa
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0: max_val = 1e-10
    heatmap = (heatmap / max_val).numpy()

    # 5. Chuyển thành ảnh màu ColorMap (Jet) để Frontend dễ nhìn
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)

    return heatmap_color