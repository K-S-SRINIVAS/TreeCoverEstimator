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
import zipfile
import torch

# from dotenv import load_dotenv

# load_dotenv()  # 👈 THIS is what reads your .env file



# -------------------------------
# --- CONFIG ---
# -------------------------------
GOOGLE_MAPS_API_KEY ='AIzaSyCULvGx6a1bD3axVuNneZV8mMfDDvM3i1Q'   # Replace with yours
print("✅ Loaded API key:", GOOGLE_MAPS_API_KEY)
PATCH_SIZE_PX = 256  # must match your training patch size

OVERLAY_DIR = "/tmp/overlays"
os.makedirs(OVERLAY_DIR, exist_ok=True)

# -------------------------------
# --- Init FastAPI ---
# -------------------------------
app = FastAPI()
from fastapi.staticfiles import StaticFiles

app.mount("/files", StaticFiles(directory=OVERLAY_DIR), name="files")


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
checkpoint = torch.load("model/unet_model.pth", map_location=device, weights_only=False)
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

# def predict_patch(image):
#     from torchvision import transforms

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
#     ])
#     x = transform(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         y = model(x)
#         y = torch.sigmoid(y)
#     mask = (y.squeeze().cpu().numpy() > 0.1).astype(np.uint8) * 255
#     return Image.fromarray(mask)

def predict_patch(image):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)
        y = torch.sigmoid(y)
        print("Model output range:", y.min().item(), y.max().item())

    mask = (y.squeeze().cpu().numpy() > 0.1).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)

    # --- Save debug images ---
    os.makedirs("debug_tiles", exist_ok=True)
    debug_name = uuid.uuid4().hex
    image.save(f"debug_tiles/{debug_name}_tile.png")
    mask_img.save(f"debug_tiles/{debug_name}_mask.png")

    # Create side-by-side comparison (tile + mask)
    combined = Image.new("RGB", (PATCH_SIZE_PX * 2, PATCH_SIZE_PX))
    combined.paste(image.resize((PATCH_SIZE_PX, PATCH_SIZE_PX)), (0, 0))
    combined.paste(mask_img.convert("RGB"), (PATCH_SIZE_PX, 0))
    combined.save(f"debug_tiles/{debug_name}_combined.png")

    return mask_img

# -------------------------------
# --- API Route ---
# -------------------------------
@app.get("/predict")
async def predict(north: float, south: float, east: float, west: float):
    zoom = 20  # fixed for good resolution
    
    area_per_pixel=0.0194
    step_deg = 0.0002976  # For Durg, zoom=20, 256x256 tile
    
    # Offset so tiles stay *inside* AOI
    start_lat = south + step_deg / 2
    end_lat = north - step_deg / 2
    start_lng = west + step_deg / 2
    end_lng = east - step_deg / 2

    lat_steps = math.ceil(abs(end_lat - start_lat) / step_deg)
    lng_steps = math.ceil(abs(end_lng - start_lng) / step_deg)

    lat_list = np.linspace(start_lat, end_lat, lat_steps)
    lng_list = np.linspace(start_lng, end_lng, lng_steps)


    print("lat",lat_list)
    print("long",lng_list)
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
    # Compute actual bounds of the stitched image
    actual_south = start_lat - step_deg / 2
    actual_north = end_lat + step_deg / 2
    actual_west  = start_lng - step_deg / 2
    actual_east  = end_lng + step_deg / 2

    # Return corrected bounds
    response = {
    "overlay_url": f"/files/{overlay_filename}",
    "bounds": [[actual_south, actual_west], [actual_north, actual_east]],
    "percent_green_cover": green_pixels*area_per_pixel
    }

    return JSONResponse(content=response)

# -------------------------------
# --- Static Files ---
# -------------------------------

# -------------------------------
# --- Run ---
# -------------------------------
# Start with: uvicorn app:app --reload --host 0.0.0.0 --port 8000
