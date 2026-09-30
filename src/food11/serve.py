import os

import mlflow
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms


app = FastAPI()


# MLflow server address
mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
)


# Load model once when API starts
model = mlflow.pyfunc.load_model(
    "models:/food11@champion"
)


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


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # Read image
    image = Image.open(
        file.file
    ).convert("RGB")


    # Apply same preprocessing as training
    image = transform(image)


    # Add batch dimension
    image = image.unsqueeze(0)


    # Model prediction
    prediction = model.predict(
        image.numpy()
    )


    # Convert output to tensor
    prediction_tensor = torch.tensor(
        prediction
    )


    # Convert logits/scores to probabilities
    probabilities = torch.softmax(
        prediction_tensor,
        dim=1
    )


    # Get predicted class
    predicted_class = int(
        probabilities.argmax(
            dim=1
        ).item()
    )


    # Get confidence between 0 and 1
    confidence = float(
        probabilities.max().item()
    )


    return {
        "category": classes[predicted_class],
        "confidence": confidence
    }