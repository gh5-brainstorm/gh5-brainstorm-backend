from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

import requests
import json
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
# 
model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

IMAGE_SHAPE = (244, 244)

model = keras.Sequential([
    hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,))
])

def extract(file, is_from_user):
  print(file)
  if (is_from_user == "yes"):
    file = Image.open(io.BytesIO(file)).convert('L').resize(IMAGE_SHAPE)
    # file = Image.close()
  else:
     file = Image.open(file).convert('L').resize(IMAGE_SHAPE)
    #  file = Image.close()

  file = np.array(file)    
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
async def create_upload_file(file: UploadFile):
    logging.info('Start comparing image uploaded by user')
    request_image = file.file.read()
    similar_image = []
    extract_img = extract(request_image, "yes")
    try:
        data = read_database()
        if len(data) > 0:
            # Transform the data to only include image_name
            for item in data:
                path = item["image_name"]
                extract(f"images/{path}", "no")

                request_image = file.file.read()
                # image_file_pil = Image.open(io.BytesIO(request_image))

                accuracy = distance.cdist([extract_img], [cat1], metric = 'cosine')[0].tolist()[0]
                if accuracy < 0.6000:
                    similar_image.append({
                        "url": item["url"],
                        "similarity_score": accuracy
                    })
                    
    except Exception as e:
        logging.error(f'Error occurred: {e}')
        raise HTTPException(status_code=500, detail="Failed to comparing image")

    logging.info('Starting file upload process')
    try:
        # upload image content
        file.file.seek(0)
        file_content = file.file.read()
        file_size = len(file_content)
        file_path = Path(IMAGES_DIR) / file.filename

        logging.info(f'Saving file to {file_path} with size {len(file_content)} bytes')

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

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
            "image_name": file.filename,
        }
        data.append(image_entry)
        write_database(data)

        curr_data = [
            {
                "id": image_id,
                "url": image_entry["url"],
                "image_name": file.filename,
                "similar_image": similar_image  # corrected spelling from 'similiar_image'
            }
        ]

        response = {
            "statusCode": 200,
            "message": "",
            "data": curr_data
        }
        return JSONResponse(
            content=response
        )
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

@app.get("/")
def get_root():
    return "Welcome to artector server-side"
