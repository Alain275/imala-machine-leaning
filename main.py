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
from contextlib import asynccontextmanager

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await startup()
    yield

app = FastAPI(
    title="IMARA AI Disease Detection Service",
    description="PlantVillage MobileNetV2 Disease Detection API",
    version="1.0.0",
    lifespan=lifespan,
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
API_KEY = os.getenv("API_KEY", "")

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
MODEL_MODE    = os.getenv("MODEL_MODE", "local").lower()
DEMO_MODE     = os.getenv("DEMO_MODE", "false").lower() == "true"
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
ACTIVE_BACKEND    = "none"


async def startup():
    global _local_model, _local_processor, ACTIVE_BACKEND

    if not API_KEY and MODEL_MODE != "demo":
        raise RuntimeError("API_KEY is required outside demo mode")

    print(f"--- IMARA AI Disease Detection Service ---")
    print(f"Startup Info:")
    print(f"  MODEL_MODE: {MODEL_MODE}")
    print(f"  USE_HF_API: {USE_HF_API}")
    print(f"  DEMO_MODE: {DEMO_MODE}")
    print(f"  HF_MODEL_PATH: {HF_MODEL_PATH}")
    hf_model_exists = os.path.exists(HF_MODEL_PATH)
    print(f"  HF_MODEL_PATH exists: {hf_model_exists}")
    if hf_model_exists:
        try:
            files = os.listdir(HF_MODEL_PATH)
            print(f"  Files in HF_MODEL_PATH: {files}")
        except Exception as e:
            print(f"  Could not list files in HF_MODEL_PATH: {e}")
    print(f"------------------------------------------")

    target_mode = MODEL_MODE

    def try_load_pytorch():
        global _local_model, _local_processor
        try:
            # Local production inference only: never contact the Hugging Face Hub.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            import json
            prep_path = os.path.join(HF_MODEL_PATH, "preprocessor_config.json")
            if os.path.exists(prep_path):
                with open(prep_path, "r") as f:
                    prep_cfg = json.load(f)
                if prep_cfg.get("image_processor_type") == "MobileNetV2FeatureExtractor":
                    prep_cfg["image_processor_type"] = "MobileNetV2ImageProcessor"
                    with open(prep_path, "w") as f:
                        json.dump(prep_cfg, f)

            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
            _local_processor = AutoImageProcessor.from_pretrained(
                HF_MODEL_PATH,
                local_files_only=True,
            )
            _local_model = AutoModelForImageClassification.from_pretrained(
                HF_MODEL_PATH,
                local_files_only=True,
            )
            _local_model.eval()
            if os.path.exists(ID2LABEL_PATH):
                with open(ID2LABEL_PATH) as f:
                    id2label = json.load(f)
                DISEASE_CLASSES[:] = [id2label[str(i)] for i in range(len(id2label))]
            return True
        except Exception as e:
            print(f"⚠️  Could not load local HF model: {e}")
            return False

    def try_load_keras():
        global _local_model
        try:
            import tensorflow as tf
            _local_model = tf.keras.models.load_model(MODEL_PATH)
            return True
        except Exception as e:
            print(f"⚠️  Could not load Keras model: {e}")
            return False

    if target_mode == "auto":
        if os.path.exists(HF_MODEL_PATH) and try_load_pytorch():
            ACTIVE_BACKEND = "local_pytorch"
        elif os.path.exists(MODEL_PATH) and try_load_keras():
            ACTIVE_BACKEND = "local_keras"
        elif HF_API_TOKEN and HF_REPO_ID:
            ACTIVE_BACKEND = "hf_api"
        elif DEMO_MODE:
            ACTIVE_BACKEND = "demo"
        else:
            raise RuntimeError("Auto mode failed: No local models found, no HF API token, and demo mode not allowed.")

    elif target_mode == "local":
        if os.path.exists(HF_MODEL_PATH) and try_load_pytorch():
            ACTIVE_BACKEND = "local_pytorch"
        elif os.path.exists(MODEL_PATH) and try_load_keras():
            ACTIVE_BACKEND = "local_keras"
        else:
            if DEMO_MODE:
                ACTIVE_BACKEND = "demo"
            else:
                raise RuntimeError(f"Startup failed: Local models missing or failed to load. HF_MODEL_PATH={HF_MODEL_PATH}")

    elif target_mode == "hf_api":
        if HF_API_TOKEN and HF_REPO_ID:
            ACTIVE_BACKEND = "hf_api"
        else:
            if DEMO_MODE:
                ACTIVE_BACKEND = "demo"
            else:
                raise RuntimeError("Startup failed: hf_api mode requires HF_API_TOKEN and HF_REPO_ID")

    elif target_mode == "demo":
        ACTIVE_BACKEND = "demo"
    else:
        raise RuntimeError(f"Unknown MODEL_MODE: {MODEL_MODE}")

    if ACTIVE_BACKEND == "local_pytorch":
        print(f"✅ Mode: Local PyTorch model from {HF_MODEL_PATH}")
    elif ACTIVE_BACKEND == "local_keras":
        print(f"✅ Mode: Local Keras model from {MODEL_PATH}")
    elif ACTIVE_BACKEND == "hf_api":
        print(f"✅ Mode: HuggingFace Inference API\n   Model: {HF_REPO_ID}")
    elif ACTIVE_BACKEND == "demo":
        print("⚠️  Running in demo mode.")


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
    model_name = "demo"
    if ACTIVE_BACKEND == "local_pytorch":
        model_name = HF_MODEL_PATH
    elif ACTIVE_BACKEND == "local_keras":
        model_name = MODEL_PATH
    elif ACTIVE_BACKEND == "hf_api":
        model_name = HF_REPO_ID

    return {
        "service": "IMARA AI Disease Detection",
        "version": "1.0.0",
        "status": "running",
        "mode": ACTIVE_BACKEND,
        "model": model_name,
        "active_backend": ACTIVE_BACKEND
    }


@app.get("/health")
def health_check():
    ready = ACTIVE_BACKEND in ["local_pytorch", "local_keras", "hf_api", "demo"]
    return {"status": "ok", "service": "imara-ml", "ready": ready, "backend": ACTIVE_BACKEND}


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

    if ACTIVE_BACKEND == "hf_api":
        disease_class, confidence = _predict_via_hf_api(contents)
    elif ACTIVE_BACKEND == "local_pytorch":
        image = Image.open(io.BytesIO(contents))
        disease_class, confidence = _predict_local_pytorch(image)
    elif ACTIVE_BACKEND == "local_keras":
        image = Image.open(io.BytesIO(contents))
        disease_class, confidence = _predict_local_keras(image)
    else:                                        # demo / fallback
        demo_mode = True
        disease_class = "Tomato___Late_blight"
        confidence = 0.87
        print("⚠️ Running in demo mode — returning mock prediction")

    disease_info = get_disease_info(disease_class)

    metadata_model = "demo"
    if ACTIVE_BACKEND == "local_pytorch":
        metadata_model = HF_MODEL_PATH
    elif ACTIVE_BACKEND == "local_keras":
        metadata_model = os.path.basename(MODEL_PATH)
    elif ACTIVE_BACKEND == "hf_api":
        metadata_model = HF_REPO_ID

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
            "model":      metadata_model,
            "classes":    len(DISEASE_CLASSES),
            "demo_mode":  demo_mode,
            "mode":       "local" if ACTIVE_BACKEND.startswith("local") else ACTIVE_BACKEND,
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
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port)
