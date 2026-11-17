import os
from contextlib import asynccontextmanager
from typing import Annotated

import mlflow
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

MODEL_URI = os.getenv("MODEL_URI", "models:/food11@champion")

classes = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# Module-level so tests can replace it with a fake model.
model = None


def load_model():
    """Load the champion model from MLflow, unless SKIP_MODEL is set
    (unit tests and the CI smoke test run without an MLflow server)."""
    global model
    if os.environ.get("SKIP_MODEL"):
        return
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    model = mlflow.pyfunc.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Same role as @app.on_event("startup"), which FastAPI now deprecates.
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(file: Annotated[UploadFile, File()]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image") from None

    # Same preprocessing as training, plus a batch dimension.
    batch = transform(image).unsqueeze(0).numpy()

    # The model returns one row of 11 raw scores (logits) per image.
    scores = torch.tensor(np.asarray(model.predict(batch), dtype=np.float32))
    probabilities = torch.softmax(scores, dim=1)

    predicted_class = int(probabilities.argmax(dim=1).item())
    confidence = float(probabilities.max().item())

    return {"category": classes[predicted_class], "confidence": confidence}
