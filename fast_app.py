# Importing the libraries 
import uvicorn ## For ASGI 
from fastapi import FastAPI , File, UploadFile
import tensorflow as tf 
import numpy as np
from PIL import Image, ImageOps
new_model = tf.keras.models.load_model('my_model.h5')

# Create the app object

app = FastAPI()


# Index route opens automatically
@app.get('/')
def index():
	return {"message":"This is a digit recognizer"}

@app.post("/predict")
def predict(file:UploadFile = File(...)):
	img = Image.open(file.filename)
	img = ImageOps.grayscale(img)
	img = img.resize((28,28))
	img_array = np.array(img)
	img_array = np.invert(img_array)
	img_array = img_array.reshape(1,28,28,1)
	y_pred = new_model.predict(img_array)
	label = np.array_str(np.argmax(y_pred,axis=1))

	return {'label':f'{label}'}


if __name__=="__main__":
	uvicorn.run(app, host='127.0.0.1',port=8000)



