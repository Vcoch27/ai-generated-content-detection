import os
import joblib
import logging
import shap
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from contextlib import asynccontextmanager

# Import các utils chuyên biệt đã xây dựng
from utils.image_processing import preprocess_for_cv, preprocess_for_cnn, encode_image_to_base64
from utils.feature_extraction import get_hybrid_vector
from utils.xai_processing import make_gradcam_heatmap
from utils.feature_analysis import log_all_feature_importances, get_local_feature_impact

# ===== CẤU HÌNH LOGGING =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Biến toàn cục để giữ model trong RAM
cnn_extractor = None
cnn_pure_full = None
pca_transformer = None
rf_classifier = None
explainer = None

# ===== ĐƯỜNG DẪN HỆ THỐNG MODELS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CNN_PATH = os.path.join(MODEL_DIR, "cnn_feature_extractor_1024.keras")
PURE_CNN_PATH = os.path.join(MODEL_DIR, "ai_detector_model_pure_cnn.keras")
PCA_PATH = os.path.join(MODEL_DIR, "hybrid_pca_transformer.joblib")
RF_PATH = os.path.join(MODEL_DIR, "ai_detector_final_model_hybrid_fusion.joblib")

# ===== QUẢN LÝ LIFESPAN (Thay cho on_event cũ) =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Phần này chạy khi startup
    global cnn_extractor, cnn_pure_full, pca_transformer, rf_classifier, explainer
    try:
        logger.info("--- Đang nạp hệ thống Hybrid Model vào RAM ---")
        cnn_extractor = load_model(CNN_PATH)
        cnn_pure_full = load_model(PURE_CNN_PATH)
        pca_transformer = joblib.load(PCA_PATH)
        rf_classifier = joblib.load(RF_PATH)
        logger.info("✓ Toàn bộ hệ thống Model (CNN, PURE CNN, PCA, RF) đã sẵn sàng!")

        log_all_feature_importances(rf_classifier)
        explainer = shap.TreeExplainer(rf_classifier)
    except Exception as e:
        logger.error(f"✗ Lỗi nạp model: {e}")

    yield  # Chờ app hoạt động

    # Phần này chạy khi shutdown (nếu cần dọn dẹp)
    logger.info("--- Đang tắt hệ thống ---")

# ===== CẤU HÌNH FASTAPI =====
app = FastAPI(
    title="AI Generated Detection - Hybrid Fusion Service",
    description="API cao cấp kết hợp CNN và Computer Vision để phát hiện ảnh AI",
    version="3.0.0",
    lifespan=lifespan
)

# CORS middleware cho Spring Boot kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ENDPOINTS =====

@app.get("/")
def read_root():
    """Kiểm tra trạng thái Service"""
    status = "Ready" if rf_classifier is not None else "Model Loading..."
    return {
        "service": "HyperID AI Detector API",
        "version": "3.0.0 (Hybrid Fusion)",
        "status": status,
        "engine": "MobileNetV2 + OpenCV + Random Forest"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Dự đoán ảnh thông qua Pipeline Hybrid V3
    """
    # 1. Kiểm tra sự sẵn sàng của hệ thống
    if rf_classifier is None:
        raise HTTPException(status_code=500, detail="Hệ thống Model chưa được nạp!")

    # 2. Kiểm tra định dạng file
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(status_code=400, detail="Định dạng ảnh không hợp lệ!")

    temp_path = f"temp_{file.filename}"

    try:
        # 3. Đọc dữ liệu byte và chuyển đổi tạm thời để xử lý
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        # 4. Gọi hàm "Vạn năng" để lấy đủ 77 thuộc tính
        features_77 = get_hybrid_vector(
            temp_path, cnn_extractor, pca_transformer,
            preprocess_for_cv, preprocess_for_cnn
        )

        # 5.1 Dự đoán Hybrid (Main logic)
        prediction = int(rf_classifier.predict(features_77)[0])
        probs = rf_classifier.predict_proba(features_77)[0]

        # 5.2 Tạo Heatmap XAI (Explanation logic)
        img_cnn = preprocess_for_cnn(temp_path)
        heatmap_img = make_gradcam_heatmap(img_cnn, cnn_pure_full)
        heatmap_base64 = encode_image_to_base64(heatmap_img)

        # 6. XÓA FILE TẠM VÀ TRẢ KẾT QUẢ
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Logic nhãn: 0 là AI_GENERATED, 1 là REAL
        result = "AI_GENERATED" if prediction == 0 else "REAL"

        # Gọi hàm phân tích với nhãn kết quả để lấy đúng ý nghĩa
        cv_analysis = get_local_feature_impact(explainer, features_77, result)

        # Độ tin cậy cho lớp được chọn
        confidence = float(probs[prediction]) * 100

        ai_prob = float(probs[0]) * 100
        real_prob = float(probs[1]) * 100

        response = {
            "filename": file.filename,
            "prediction": result,
            "confidence": round(confidence, 2),
            "ai_probability": round(ai_prob, 2),
            "real_probability": round(real_prob, 2),
            "heatmap_base64": heatmap_base64,
            "cv_analysis": cv_analysis,
            "status": "success"
        }

        logger.info(f"Prediction: {result} ({confidence:.2f}%) for {file.filename}")
        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== CHẠY SERVER =====
if __name__ == "__main__":
    import uvicorn

    # Chạy trên port 8000 để Spring Boot gọi tới
    uvicorn.run(app, host="0.0.0.0", port=8000)