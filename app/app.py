from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

import requests
import json
import base64
from pathlib import Path
import logging
import os

import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
import numpy as np
import tf_keras as keras
from scipy.spatial import distance
import io

model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

IMAGE_SHAPE = (244, 244)

model = keras.Sequential([
    hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,))
])

def extract(file, is_from_user):
  if (is_from_user == "yes"):
    file = Image.open(io.BytesIO(file)).convert('L').resize(IMAGE_SHAPE)
  else:
     file = Image.open(file).convert('L').resize(IMAGE_SHAPE)

#   file = np.array(file)    
  file = np.stack((file,)*3, axis=-1)
  file = np.array(file)/255.0

  embedding = model.predict(file[np.newaxis, ...])

  vgg16_feature_np = np.array(embedding)
  flattended_feature = vgg16_feature_np.flatten()

  return flattended_feature

app = FastAPI()

logging.basicConfig(level=logging.INFO)

DATABASE_FILE = 'database.json'
IMAGES_DIR = 'images'

# Ensure the images directory exists
os.makedirs(IMAGES_DIR, exist_ok=True)

# Ensure the database file exists
if not os.path.exists(DATABASE_FILE):
    with open(DATABASE_FILE, 'w') as file:
        json.dump([], file)

def read_database():
    with open(DATABASE_FILE, 'r') as file:
        return json.load(file)

def write_database(data):
    with open(DATABASE_FILE, 'w') as file:
        json.dump(data, file, indent=4)


@app.post("/upload")
async def create_upload_file(file: UploadFile = File(...)):
    logging.info('Start comparing image uploaded by user')
    try:
        cat1 = extract('images/2560px-A-Cat.jpg', "no")

        request_image = file.file.read()
        # image_file_pil = Image.open(io.BytesIO(request_image))
        extract_img = extract(request_image, "yes")

        result = distance.cdist([extract_img], [cat1], metric = 'cosine')[0]
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise HTTPException(status_code=500, detail="Failed to comparing image")

    logging.info('Starting file upload process')
    try:
        # upload image content
        file_content = await file.read()
        file_path = Path(IMAGES_DIR) / file.filename
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        data = read_database()
        image_id = len(data) + 1
        ngrok_url = get_ngrok_url()
        if not ngrok_url:
            raise HTTPException(status_code=500, detail="Failed to get Ngrok URL")
        
        image_entry = {
            "id": image_id,
            "url": f"{ngrok_url}/images/{file.filename}",
<<<<<<< HEAD
=======
            "base64": base64_content  # Store base64 content in database for future use,
            # "similiarity_score": result
>>>>>>> cc2f734d912076154320adbe8aa473f2ca68a02d
        }
        data.append(image_entry)
        write_database(data)

        return {"status": "ok", "id": image_id, "url": image_entry["url"], 'similirarity_score': result.tolist()}
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise HTTPException(status_code=500, detail="Failed to upload file")

def get_ngrok_url():
    try:
        # Query the local Ngrok API
        response = requests.get('http://localhost:4040/api/tunnels')
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Parse the JSON response
        tunnels = response.json().get('tunnels', [])
        for tunnel in tunnels:
            if tunnel.get('proto') == 'https':
                return tunnel.get('public_url')

    except requests.RequestException as e:
        print(f"Error fetching Ngrok URL: {e}")
        return None

# Fetch and print the Ngrok HTTPS URL
ngrok_url = get_ngrok_url()

@app.get("/images/{filename}")
def get_image(filename: str):
    file_path = Path(IMAGES_DIR) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


@app.get("/images")
def get_all_images():
    data = read_database()
    response = {
        "statusCode": 200,
        "message": "",
        "data": data
    }
    return JSONResponse(content=response)
