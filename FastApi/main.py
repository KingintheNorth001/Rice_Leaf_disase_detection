from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()
MODEL = tf.keras.models.load_model("D:\\deep_learning\\Rice_leaf_disease_detection_DL\\saved_models\\3.keras")
Class_name = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']


@app.get("/ping")
async def ping():
    return "hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array,0)
    predictions = MODEL.predict(img_array)
    predicted_class = Class_name[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host = 'localhost', port=8000)
