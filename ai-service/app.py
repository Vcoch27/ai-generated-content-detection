import os
import io
import cv2
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from utils.feature_extractor import extract_features, create_feature_dataframe

# ===== CẤU HÌNH LOGGING =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CẤU HÌNH FASTAPI =====
app = FastAPI(
    title="AI Generated Content Detection Service",
    description="REST API để phát hiện ảnh được sinh ra bởi AI",
    version="1.0.0"
)

# Thêm CORS middleware để cho phép Spring Boot gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== TẢI MODEL =====
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "ai_detector_model_v2.joblib")

try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"✗ Failed to load model: {e}")
    model = None


# ===== ENDPOINTS =====

@app.get("/")
def read_root():
    """Root endpoint - kiểm tra service có hoạt động không"""
    return {
        "status": "running",
        "service": "AI Generated Content Detection",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model": model_status
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint dự đoán ảnh.

    - Nhận: file ảnh (jpg, jpeg, png, bmp)
    - Trả: JSON chứa prediction và confidence
    """

    # Kiểm tra model
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Kiểm tra file
    try:
        filename = file.filename
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            raise HTTPException(status_code=400, detail="Invalid file format. Accepted: jpg, jpeg, png, bmp")

        # Đọc file ảnh
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        # Convert bytes → OpenCV image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        logger.info(f"Processing image: {filename} (shape: {img.shape})")

        # Trích xuất đặc trưng
        features = extract_features(img)
        if features is None:
            raise HTTPException(status_code=500, detail="Failed to extract features from image")

        # Chuyển thành DataFrame
        features_df = create_feature_dataframe(features)
        if features_df is None:
            raise HTTPException(status_code=500, detail="Failed to create feature dataframe")

        # Dự đoán
        prediction = model.predict(features_df)[0]
        probs = model.predict_proba(features_df)[0]

        # Xác định kết quả
        if prediction == 1:
            result = "AI_GENERATED"
            confidence = float(probs[1]) * 100
        else:
            result = "REAL"
            confidence = float(probs[0]) * 100

        response = {
            "filename": filename,
            "prediction": result,
            "confidence": round(confidence, 2),
            "ai_probability": round(float(probs[1]) * 100, 2),
            "real_probability": round(float(probs[0]) * 100, 2),
            "status": "success"
        }

        logger.info(f"Prediction: {result} ({confidence:.2f}%) for {filename}")
        return JSONResponse(content=response, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Endpoint dự đoán nhiều ảnh cùng lúc.

    - Nhận: danh sách file ảnh
    - Trả: danh sách kết quả JSON
    """

    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")

    results = []

    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": "Failed to decode image"
                })
                continue

            features = extract_features(img)
            if features is None:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": "Failed to extract features"
                })
                continue

            features_df = create_feature_dataframe(features)
            prediction = model.predict(features_df)[0]
            probs = model.predict_proba(features_df)[0]

            result_label = "AI_GENERATED" if prediction == 1 else "REAL"
            confidence = (float(probs[1]) if prediction == 1 else float(probs[0])) * 100

            results.append({
                "filename": file.filename,
                "prediction": result_label,
                "confidence": round(confidence, 2),
                "status": "success"
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })

    return {
        "total": len(files),
        "processed": len([r for r in results if r["status"] == "success"]),
        "results": results
    }


# ===== CHẠY SERVER =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
