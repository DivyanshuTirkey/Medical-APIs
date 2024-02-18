import os
from io import BytesIO
import tensorflow as tf
from fastapi.logger import logger
from fastapi import FastAPI, UploadFile, File

model_posco = tf.keras.models.load_model("models/poscogen/PCOSGen-train.h5")
posco_pred = ["Healthy", "Unhealthy"]

app = FastAPI()
@app.get('/')
def check_status():
    return ["The server is Running"]

@app.post('/predict_pcos')
async def predict_posco(image: UploadFile = File(...)):
    contents = await image.read()
    with open(image.filename, 'wb') as f:
        f.write(contents)
    contents = tf.io.read_file(image.filename)
    os.remove(image.filename)
    contents = tf.io.decode_image(contents)
    contents = tf.expand_dims(contents, axis = 0)
    contents = tf.image.resize(contents, (180, 180))
    prediction = tf.math.argmax(model_posco.predict(contents), 1)
    label = posco_pred[prediction[0]]
    return {"prediction": label}