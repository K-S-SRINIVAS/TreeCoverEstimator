# --- app.py ---

import os
import io
import math
import uuid
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import torch
import segmentation_models_pytorch as smp
# from dotenv import load_dotenv

# load_dotenv()  # 👈 THIS is what reads your .env file



# -------------------------------
# --- CONFIG ---
# -------------------------------
GOOGLE_MAPS_API_KEY ='AIzaSyCULvGx6a1bD3axVuNneZV8mMfDDvM3i1Q'   # Replace with yours
print("✅ Loaded API key:", GOOGLE_MAPS_API_KEY)
PATCH_SIZE_PX = 256  # must match your training patch size

OVERLAY_DIR = "static/overlays"
os.makedirs(OVERLAY_DIR, exist_ok=True)

# -------------------------------
# --- Init FastAPI ---
# -------------------------------
app = FastAPI()

# CORS for local dev:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Or specify your React origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# --- Load Model ---
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)
checkpoint = torch.load("model/unet_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# -------------------------------
# --- Utils ---
# -------------------------------
def fetch_tile(lat, lng, zoom, size=PATCH_SIZE_PX):
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lng}&zoom={zoom}&size={size}x{size}&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        raise Exception(f"Failed to fetch tile: {response.text}")

def predict_patch(image):
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        y = model(x)
        y = torch.sigmoid(y)
    mask = (y.squeeze().cpu().numpy() > 0.1).astype(np.uint8) * 255
    return Image.fromarray(mask)

# -------------------------------
# --- API Route ---
# -------------------------------
@app.get("/predict")
async def predict(north: float, south: float, east: float, west: float):
    zoom = 19  # fixed for good resolution

    # --- 1) Compute center & patch grid ---
    lat_steps = math.ceil(abs(north - south) / 0.0005)  # adjust this factor to match zoom level meters
    lng_steps = math.ceil(abs(east - west) / 0.0005)
    lat_list = np.linspace(south, north, lat_steps)
    lng_list = np.linspace(west, east, lng_steps)

    # --- 2) Fetch tiles and predict ---
    stitched_image = Image.new("L", (lng_steps * PATCH_SIZE_PX, lat_steps * PATCH_SIZE_PX))
    green_pixels = 0
    total_pixels = stitched_image.width * stitched_image.height

    for i, lat in enumerate(lat_list):
        for j, lng in enumerate(lng_list):
            tile = fetch_tile(lat, lng, zoom)
            mask = predict_patch(tile)
            stitched_image.paste(mask, (j * PATCH_SIZE_PX, i * PATCH_SIZE_PX))
            green_pixels += np.count_nonzero(np.array(mask) > 127)

    # --- 3) Save stitched mask overlay ---
    overlay_filename = f"{uuid.uuid4().hex}.png"
    overlay_path = os.path.join(OVERLAY_DIR, overlay_filename)
    stitched_image.save(overlay_path)

    # --- 4) Calculate % green cover ---
    percent_green = round((green_pixels / total_pixels) * 100, 2)

    # --- 5) Return response ---
    response = {
        "overlay_url": f"/static/overlays/{overlay_filename}",
        "bounds": [[south, west], [north, east]],
        "percent_green_cover": percent_green
    }

    return JSONResponse(content=response)

# -------------------------------
# --- Static Files ---
# -------------------------------
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------
# --- Run ---
# -------------------------------
# Start with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
