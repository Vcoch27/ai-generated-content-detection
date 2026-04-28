# Hướng Dẫn Chạy AI Detection Service

## 📋 Yêu Cầu
- Python 3.12+
- Windows/Linux/Mac

## 🚀 Cách Chạy

### 1. Setup Môi Trường

```bash
# Di chuyển vào thư mục ai-service
cd ai-service

# Tạo virtual environment
python -m venv .venv

# Kích hoạt (Windows)
.\.venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source .venv/bin/activate
```

### 2. Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

### 3. Chạy Server

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     ✓ Model loaded successfully
```

### 4. Truy Cập API

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📡 Sử Dụng API

### Test Upload Ảnh

```bash
# Terminal 1: Chạy server
python -m uvicorn app:app --port 8000

# Terminal 2: Upload ảnh
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
```

### Response Mẫu

```json
{
  "filename": "image.jpg",
  "prediction": "AI_GENERATED",
  "confidence": 78.02,
  "ai_probability": 78.02,
  "real_probability": 21.98,
  "status": "success"
}
```

## 📁 Cấu Trúc Dự Án

```
ai-service/
├── app.py                      # FastAPI application
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── model/
│   └── ai_detector_model_v2.joblib  # ML Model (27.5MB)
└── utils/
    ├── feature_extractor.py    # 27 features extraction
    └── __init__.py
```

## 🔧 Troubleshoot

**Port 8000 đã bị chiếm?**
```bash
python -m uvicorn app:app --port 8001
```

**Model không load?**
```bash
# Check file model tồn tại
ls -la model/ai_detector_model_v2.joblib
```

**Import error?**
```bash
pip install -r requirements.txt
```

## 🎯 Chức Năng

- ✅ Predict single image
- ✅ Batch processing
- ✅ Health check
- ✅ Interactive docs (Swagger UI)
- ✅ CORS enabled (cho Spring Boot)

---

**Liên Hệ**: Xem README.md để chi tiết hơn
