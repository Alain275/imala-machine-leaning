from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import numpy as np
from PIL import Image
import io   
import os   
import sys
import json
import base64
import urllib.request
import urllib.error
from dotenv import load_dotenv

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

app = FastAPI(
    title="IMARA AI Disease Detection Service",
    description="PlantVillage MobileNetV2 Disease Detection API",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:5000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Authentication
API_KEY = os.getenv("API_KEY", "imara-ai-key-2024-secure")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# PlantVillage Disease Classes (38 classes)
DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Disease Treatment Recommendations
DISEASE_RECOMMENDATIONS = {
    "Late_blight": {
        "symptoms": "Dark brown to black lesions on leaves and stems, white mold on undersides",
        "treatment": "1. Remove and destroy infected plants immediately\n2. Apply copper-based fungicide\n3. Improve air circulation\n4. Avoid overhead watering",
        "prevention": "Plant resistant varieties, ensure proper spacing, use drip irrigation"
    },
    "Early_blight": {
        "symptoms": "Brown spots with concentric rings on older leaves",
        "treatment": "1. Apply fungicide containing chlorothalonil\n2. Remove infected leaves\n3. Mulch around plants",
        "prevention": "Crop rotation, avoid overhead watering, maintain plant health"
    },
    "Bacterial_spot": {
        "symptoms": "Small dark spots on leaves and fruits",
        "treatment": "1. Apply copper-based bactericide\n2. Remove severely infected plants\n3. Disinfect tools between uses",
        "prevention": "Use disease-free seeds, practice crop rotation, avoid working with wet plants"
    },
    "Powdery_mildew": {
        "symptoms": "White powdery coating on leaves",
        "treatment": "1. Apply sulfur or neem oil spray\n2. Remove heavily infected leaves\n3. Increase air circulation",
        "prevention": "Plant in full sun, avoid overcrowding, water at base of plants"
    },
    "healthy": {
        "symptoms": "No disease symptoms detected",
        "treatment": "Continue good agricultural practices",
        "prevention": "Maintain proper nutrition, adequate watering, and pest monitoring"
    }
}

# ── Runtime configuration ─────────────────────────────────────────────────────
USE_HF_API    = os.getenv("USE_HF_API", "false").lower() == "true"
HF_API_TOKEN  = os.getenv("HF_API_TOKEN", "")
HF_REPO_ID    = os.getenv("HF_REPO_ID",
                           "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
HF_API_URL    = f"https://api-inference.huggingface.co/models/{HF_REPO_ID}"

MODEL_PATH    = os.getenv("MODEL_PATH",    "./models/plantvillage_mobilenetv2.h5")
HF_MODEL_PATH = os.getenv("HF_MODEL_PATH", "./models/hf_model")
ID2LABEL_PATH = "./models/id2label.json"

# Lazy-loaded local model (Keras or PyTorch)
_local_model      = None
_local_processor  = None


@app.on_event("startup")
async def startup():
    global _local_model, _local_processor

    # ── Mode 1: HuggingFace Inference API ────────────────────────────────────
    if USE_HF_API:
        if HF_API_TOKEN:
            print(f"✅ Mode: HuggingFace Inference API")
            print(f"   Model: {HF_REPO_ID}")
        else:
            print("⚠️  USE_HF_API=true but HF_API_TOKEN is empty!")
            print("   Run  python download_model.py  and choose option 1.")
        return

    # ── Mode 2: Local HuggingFace / PyTorch model ────────────────────────────
    if os.path.exists(HF_MODEL_PATH):
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
            _local_processor = AutoImageProcessor.from_pretrained(HF_MODEL_PATH)
            _local_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_PATH)
            _local_model.eval()
            if os.path.exists(ID2LABEL_PATH):
                with open(ID2LABEL_PATH) as f:
                    id2label = json.load(f)
                DISEASE_CLASSES[:] = [id2label[str(i)] for i in range(len(id2label))]
            print(f"✅ Mode: Local PyTorch model from {HF_MODEL_PATH}")
            return
        except Exception as e:
            print(f"⚠️  Could not load local HF model: {e}")

    # ── Mode 3: Local Keras .h5 model ────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            _local_model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✅ Mode: Local Keras model from {MODEL_PATH}")
            return
        except Exception as e:
            print(f"⚠️  Could not load Keras model: {e}")

    # ── Mode 4: Demo mode ─────────────────────────────────────────────────────
    print("⚠️  No model configured.  Running in demo mode.")
    print("   Run  python download_model.py  to set up a model.")


# ─────────────────────────────────────────────────────────────────────────────
def _predict_via_hf_api(image_bytes: bytes) -> tuple[str, float]:
    """Call HuggingFace Inference API and return (disease_class, confidence)."""
    req = urllib.request.Request(
        HF_API_URL,
        data=image_bytes,
        headers={
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/octet-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            results = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        if e.code == 503:
            raise HTTPException(
                status_code=503,
                detail=f"Model is loading on HuggingFace servers, retry in ~20 seconds. ({body})"
            )
        raise HTTPException(status_code=502, detail=f"HuggingFace API error {e.code}: {body}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace API unreachable: {e}")

    # HF returns [{label: "...", score: 0.92}, ...]
    if not results or not isinstance(results, list):
        raise HTTPException(status_code=502, detail="Unexpected response from HuggingFace API")

    top = results[0]
    return top["label"], float(top["score"])


def _predict_local_pytorch(image: Image.Image) -> tuple[str, float]:
    import torch
    image = image.convert("RGB")
    inputs = _local_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = _local_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].numpy()
    idx = int(np.argmax(probs))
    return DISEASE_CLASSES[idx], float(probs[idx])


def _predict_local_keras(image: Image.Image) -> tuple[str, float]:
    arr = np.array(image.convert("RGB").resize((224, 224))) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = _local_model.predict(arr)
    idx = int(np.argmax(preds[0]))
    return DISEASE_CLASSES[idx], float(preds[0][idx])


def get_disease_info(disease_name: str) -> dict:
    parts = disease_name.split("___")
    crop    = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

    rec_key = None
    for key in DISEASE_RECOMMENDATIONS:
        if key.lower() in disease.lower():
            rec_key = key
            break

    rec = DISEASE_RECOMMENDATIONS.get(rec_key, {
        "symptoms": "Consult with an agronomist for detailed diagnosis",
        "treatment": "Professional agronomist review recommended",
        "prevention": "Maintain good agricultural practices"
    })
    return {"crop": crop, "disease": disease, **rec}


# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    mode = (
        "hf_api"   if USE_HF_API and HF_API_TOKEN else
        "local_pt" if _local_processor is not None else
        "local_keras" if _local_model is not None and not USE_HF_API else
        "demo"
    )
    return {
        "service": "IMARA AI Disease Detection",
        "version": "1.0.0",
        "status": "running",
        "mode": mode,
        "model": HF_REPO_ID if USE_HF_API else MODEL_PATH,
    }


@app.get("/health")
def health_check():
    ready = USE_HF_API and bool(HF_API_TOKEN) or _local_model is not None
    return {"status": "healthy", "ready": ready}


@app.post("/api/detect")
async def detect_disease(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Detect plant disease from an uploaded image.

    Returns disease, confidence, crop type, symptoms, treatment and prevention advice.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()

    # ── Route to correct inference backend ───────────────────────────────────
    demo_mode = False

    if USE_HF_API and HF_API_TOKEN:
        disease_class, confidence = _predict_via_hf_api(contents)
    elif _local_processor is not None:          # local PyTorch (HF model)
        image = Image.open(io.BytesIO(contents))
        disease_class, confidence = _predict_local_pytorch(image)
    elif _local_model is not None:              # local Keras .h5
        image = Image.open(io.BytesIO(contents))
        disease_class, confidence = _predict_local_keras(image)
    else:                                        # demo / fallback
        demo_mode = True
        disease_class = "Tomato___Late_blight"
        confidence = 0.87
        print("⚠️ Running in demo mode — returning mock prediction")

    disease_info = get_disease_info(disease_class)

    return {
        "success": True,
        "prediction": {
            "disease":    disease_info["disease"],
            "confidence": round(confidence * 100, 2),
            "crop":       disease_info["crop"],
            "status":     "pending_review",
            "warning":    "⚠️ AI Prediction - Requires Agronomist Verification",
        },
        "details": {
            "symptoms":   disease_info["symptoms"],
            "treatment":  disease_info["treatment"],
            "prevention": disease_info["prevention"],
        },
        "metadata": {
            "model":      HF_REPO_ID if USE_HF_API else os.path.basename(MODEL_PATH),
            "classes":    len(DISEASE_CLASSES),
            "demo_mode":  demo_mode,
            "mode":       "hf_api" if USE_HF_API else ("local" if not demo_mode else "demo"),
        }
    }


@app.get("/api/diseases")
def get_supported_diseases(api_key: str = Depends(verify_api_key)):
    """Get list of supported disease classes."""
    return {
        "success": True,
        "total_classes": len(DISEASE_CLASSES),
        "diseases": DISEASE_CLASSES
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
