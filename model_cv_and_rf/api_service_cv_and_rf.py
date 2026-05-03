from scipy.stats import skew
from skimage.feature import graycoprops
from skimage.feature import graycomatrix
from fastapi import FastAPI, UploadFile, File
import joblib
import cv2
import numpy as np
import io

# Khởi tạo API
app = FastAPI()

# Load models (đảm bảo file .joblib nằm cùng thư mục)
model = joblib.load('ai_detector_model_cv_and_rf.joblib')

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

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # 1. Đọc dữ liệu ảnh từ request (Spring Boot gửi sang)
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Không thể đọc được ảnh"}

        # 2. Trích xuất đặc trưng
        features = extract_features(img)

        # 3. Dự đoán bằng models
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]

        # 4. Trả kết quả về dạng JSON
        label = "AI-GENERATED" if prediction == 1 else "REAL-IMAGE"
        confidence = float(max(probabilities) * 100)

        return {
            "status": "success",
            "prediction": label,
            "confidence": f"{confidence:.2f}%"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Để chạy: uvicorn api_service:app --port 5000 --reload