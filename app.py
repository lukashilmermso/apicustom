# app.py
import subprocess
import uuid
from flask import Flask, request, jsonify, send_file, render_template
import requests
from werkzeug.utils import secure_filename
import os
import ffmpeg
from PIL import Image, ExifTags
import io
import logging
import base64
#from ultralytics import YOLO
import numpy as np
import cv2


def create_app():
    app = Flask(__name__, static_folder='uploads', static_url_path='/uploads')
    app.config['UPLOAD_FOLDER'] = '/app/uploads/'
    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    # Other setup code...
    return app


app = create_app()

app.logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
app.logger.addHandler(stream_handler)

#model = YOLO('bestClass.pt')

@app.route('/', methods=['GET'])
def homepage():
    return "Homepage"


@app.route('/hello', methods=['GET'])
def hello():
    return "Hello"

@app.route('/image', methods=['POST'])
def image():
    if 'file' not in request.files:
        app.logger.error("No file part in the request.")
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        # Save the uploaded file to the 'uploads' folder
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', file.filename)
        file.save(filepath)

        # Open the image using PIL
        image = Image.open(filepath)
       
        # Process the image using YOLO
        results = model(image)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()

        # Get the name of the object with the highest probability
        best_prediction = names_dict[np.argmax(probs)]

        # Pass YOLO results to the template for display
        return best_prediction

def process_image(file):
    # Open the image using PIL
    image = Image.open(file)
    if hasattr(image, '_getexif'):
        exif = dict(image._getexif().items())
        if ExifTags.TAG_ORIENTATION in exif:
            orientation = exif[ExifTags.TAG_ORIENTATION]
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    # Convert PIL image to OpenCV format
    opencv_image = np.array(image)
    opencv_image = opencv_image[:, :, ::-1]

  


    
    # Turn the entire image red
    opencv_image[:, :, 0] = 0  # Set blue channel to 0
    opencv_image[:, :, 1] = 0  # Set green channel to 0


    
    # Convert OpenCV image back to PIL format
    modified_pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    
    # Save the modified image as a temporary file
    modified_image_io = io.BytesIO()
    modified_pil_image.save(modified_image_io, format='JPEG')
    modified_image_io.seek(0)
    
    return modified_image_io

@app.route('/redImage', methods=['POST'])
def red_image():
    if 'file' not in request.files:
        app.logger.error("No file part in the request.")
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    modified_image_io = process_image(file)

    return send_file(modified_image_io, mimetype='image/JPEG')


