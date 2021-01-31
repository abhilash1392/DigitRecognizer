# Importing the Libraries:
from flask import Flask, request
import tensorflow as tf 
import numpy as np
from PIL import Image, ImageOps
import os

new_model = tf.keras.models.load_model('my_model.h5')

app=Flask(__name__)

@app.route('/')
def index():
	return '<h1>Welcome to the digit recognizer</h1>'

@app.route('/predict', methods=['POST'])
def predict():
	img = Image.open(request.files.get('file'))
	img = ImageOps.grayscale(img)
	img = img.resize((28,28))
	img_array = np.array(img)
	img_array = np.invert(img_array)
	img_array = img_array.reshape(1,28,28,1)
	y_pred = new_model.predict(img_array)
	label = np.array_str(np.argmax(y_pred,axis=1))

	return label[1]
	

if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)