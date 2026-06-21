from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
from dotenv import load_dotenv
from simple_predictor import predict_disease, DISEASE_CLASSES

load_dotenv()

app = FastAPI(
    title="IMARA AI Disease Detection Service",
    description="Plant Disease Detection API with Image Analysis",
    version="2.0.0"
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
    "Common_rust": {
        "symptoms": "Orange-brown pustules on leaves",
        "treatment": "1. Apply fungicide\n2. Remove infected leaves\n3. Ensure good air flow",
        "prevention": "Plant resistant varieties, avoid wet conditions"
    },
    "healthy": {
        "symptoms": "No disease symptoms detected",
        "treatment": "Continue good agricultural practices",
        "prevention": "Maintain proper nutrition, adequate watering, and pest monitoring"
    }
}


def get_disease_info(disease_name: str) -> dict:
    """Extract crop and disease info with recommendations"""
    parts = disease_name.split("___")
    crop = parts[0].replace("_", " ")
    disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"

    # Find matching recommendation
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


@app.get("/")
def root():
    return {
        "service": "IMARA AI Disease Detection",
        "version": "2.0.0",
        "status": "running",
        "mode": "image_analysis",
        "supported_classes": len(DISEASE_CLASSES)
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "ready": True,
        "mode": "image_analysis"
    }


@app.post("/api/detect")
async def detect_disease(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Detect plant disease from uploaded image using image analysis
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Predict using simple predictor
        disease_class, confidence = predict_disease(image)
        disease_info = get_disease_info(disease_class)
        
        return {
            "success": True,
            "prediction": {
                "disease": disease_info["disease"],
                "confidence": round(confidence * 100, 2),
                "crop": disease_info["crop"],
                "status": "pending_review",
                "warning": "⚠️ AI Prediction - Requires Agronomist Verification",
            },
            "details": {
                "symptoms": disease_info["symptoms"],
                "treatment": disease_info["treatment"],
                "prevention": disease_info["prevention"],
            },
            "metadata": {
                "model": "image_analysis_v2",
                "classes": len(DISEASE_CLASSES),
                "demo_mode": False,
                "mode": "image_analysis",
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/api/diseases")
def get_supported_diseases(api_key: str = Depends(verify_api_key)):
    """Get list of supported disease classes"""
    return {
        "success": True,
        "total_classes": len(DISEASE_CLASSES),
        "diseases": DISEASE_CLASSES
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
