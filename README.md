---
title: AI Generated Content Detection API
sdk: docker
app_port: 7860
---

# AI Generated Content Detection API

This Space runs the FastAPI service from `ai-service` and exposes:

- `GET /` for health/service status
- `POST /predict` for image prediction

## Local Run

```bash
docker build -t ai-detector-space .
docker run -p 7860:7860 ai-detector-space
```

Then open `http://localhost:7860/docs`.
