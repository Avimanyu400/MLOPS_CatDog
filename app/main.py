import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pathlib import Path
import numpy as np

from app.schemas import ImageInferenceRequest, InferenceResponse

# Initialize FastAPI app
app = FastAPI(
    title="CNN Image Classification API",
    description="Deployment of PyTorch SimpleCNN model using FastAPI and Pydantic",
    version="1.0.0"
)

# Define Model Architecture (matching Source 1)
class SimpleCNN(torch.nn.Module):
    def __init__(self, img_size=64):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(2, 2)
        reduced = img_size // 8
        self.fc1 = torch.nn.Linear(64 * reduced * reduced, 128)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Globals for model and metadata
model = None
classes = []
img_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_model():
    global model, classes, img_size
    model_path = Path("models/cnn_baseline.pt")
    
    if not model_path.exists():
        raise RuntimeError(f"Model weights not found at {model_path}. Train the model first using train_cnn.py[cite: 1].")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    img_size = checkpoint.get("img_size", 64)
    classes = checkpoint.get("classes", ["Class 0", "Class 1"])
    
    # Initialize and load weights
    model = SimpleCNN(img_size=img_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device} with image size {img_size} and classes {classes}")

# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "img_size": img_size,
        "classes": classes
    }

@app.post("/predict", response_model=InferenceResponse)
def predict(payload: ImageInferenceRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        # Convert incoming list to numpy array and tensor
        img_array = np.array(payload.image, dtype=np.float32)
        
        # Validate shape: Expected [3, img_size, img_size]
        expected_shape = (3, img_size, img_size)
        if img_array.shape != expected_shape:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid image shape {img_array.shape}. Expected {expected_shape}"
            )
        
        # If values are given in range [0, 255], normalize them
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
            
        # Add batch dimension -> [1, 3, H, W]
        input_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probabilities))
            
        return {
            "predicted_class_index": pred_idx,
            "predicted_class_name": str(classes[pred_idx]),
            "probabilities": probabilities.tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))