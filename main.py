# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://plant-leaf-health.vercel.app", "http://localhost:5173"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "best_model.keras"   
IMG_SIZE = (128, 128)            

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

@app.get("/")
def root():
    return {"status": "ok", "message": "FastAPI is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0).astype(np.float32)
        pred = model.predict(arr)[0][0]
        label = "Unhealthy" if pred > 0.5 else "Healthy"
        return {"label": label}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
