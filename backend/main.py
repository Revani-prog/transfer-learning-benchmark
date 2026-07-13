from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io
import time

app = FastAPI(title="Transfer Learning Benchmark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "../results/metrics/results.json")
MODELS_PATH  = os.path.join(os.path.dirname(__file__), "../results/models")

NUM_CLASSES = 102
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_results():
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def get_metrics():
    return load_results()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    predictions = {}

    model_configs = {
        "ResNet50":       ("ResNet50.pth",       "resnet50"),
        "EfficientNet-B0": ("EfficientNet-B0.pth", "efficientnet"),
        "MobileNetV2":    ("MobileNetV2.pth",    "mobilenet"),
        "VGG16":          ("VGG16.pth",          "vgg16"),
    }

    for model_name, (filename, arch) in model_configs.items():
        model_path = os.path.join(MODELS_PATH, filename)
        if not os.path.exists(model_path):
            predictions[model_name] = {"error": "Model not found"}
            continue

        # Load model
        if arch == "resnet50":
            model = models.resnet50()
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif arch == "efficientnet":
            model = models.efficientnet_b0()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        elif arch == "mobilenet":
            model = models.mobilenet_v2()
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        elif arch == "vgg16":
            model = models.vgg16()
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        start = time.time()
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        inference_ms = round((time.time() - start) * 1000, 2)

        predictions[model_name] = {
            "class_id": predicted.item(),
            "confidence": round(confidence.item() * 100, 2),
            "inference_ms": inference_ms
        }

    return {"predictions": predictions}