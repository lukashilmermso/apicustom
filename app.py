# app.py
import subprocess
import uuid
from flask import Flask, request, jsonify, send_file, render_template
import requests
from werkzeug.utils import secure_filename
import os
import ffmpeg
from PIL import Image, ImageDraw
import io
import logging
import base64
from ultralytics import YOLO
import numpy as np
import cv2


def create_app():
    app = Flask(__name__, static_folder='uploads', static_url_path='/uploads')
    app.config['UPLOAD_FOLDER'] = '/app/uploads/'
    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    app.config['OUTPUT_FOLDER'] = '/app/output_folder/'
    output_folder = app.config['OUTPUT_FOLDER']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Delete everything in output_folder if it exists
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    
    # Other setup code...
    return app


app = create_app()

app.logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
app.logger.addHandler(stream_handler)

#model = YOLO('bestBBOXES.pt')
model = YOLO('bestALLCLASSES.pt')
model1 = YOLO('bestClass.pt')
multipleClasses = {0: "Form_1"}
oneClass = {1: "Form_2", 2: "Form_3", 3: "Form_4", 4: "Form_5"}
combinedClasses = {**multipleClasses, **oneClass}

@app.route('/', methods=['GET'])
def homepage():
    return "Homepagee"


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
        if 274 in exif:  # Attribute code for orientation
            orientation = exif[274]
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    
    opencv_image = np.array(image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    results = model.predict(source=opencv_image, show=False, conf=0.75)
    color = (0, 255, 0)  # Color of the rectangle (in BGR format)
    thickness = 2  # Thickness of the rectangle's border
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for i, box in enumerate(boxes):
            type = combinedClasses[box.cls[0].astype(int)]
            if box.conf[0] > 0.75:
                if type in multipleClasses.values():
        
                    r = box.xyxy[0].astype(int)
        
                    top_left = (r[0], r[1])
                    bottom_right = (r[2], r[3])

                    crop = opencv_image[r[1] - 20:r[3] + 20, r[0]:r[2]]
                    basepath = os.path.dirname(__file__)
                    output_path = os.path.join(basepath, 'output_folder', str(i) + ".jpg")

                    cv2.imwrite(output_path, crop)

                    results = model1(output_path)

                    names_dict = results[0].names

                    probs = results[0].probs.data.tolist()
                    
                    label_text = "Form_1_" + names_dict[np.argmax(probs)] + "(Conf: " + str(round(box.conf[0], 2)) + ")"
                    cv2.rectangle(opencv_image, top_left, bottom_right, color, thickness)
    
                    label_position = (top_left[0], top_left[1] - 10)  # Just above the rectangle
                    
                    # Define the font settings
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 3
                    font_color = (0, 255, 0)
                    font_thickness = 10
                    
                    # Add the custom label to the image
                    cv2.putText(opencv_image, label_text, label_position, font, font_scale, font_color, font_thickness)
                else:
                    r = box.xyxy[0].astype(int)
                    
                    top_left = (r[0], r[1])
                    bottom_right = (r[2], r[3])
                    label_text = "Form_2" + "(Conf: " + str(round(box.conf[0], 2)) + ")"
                    cv2.rectangle(opencv_image, top_left, bottom_right, color, thickness)
    
                    label_position = (top_left[0], top_left[1] - 10)  # Just above the rectangle
                    
                    # Define the font settings
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 3
                    font_color = (0, 255, 0)
                    font_thickness = 10
                    
                    # Add the custom label to the image
                    cv2.putText(opencv_image, label_text, label_position, font, font_scale, font_color, font_thickness)
            else:
                break
    
    # Draw the rectangle on the image
    #cv2.rectangle(opencv_image, top_left, bottom_right, color, thickness)
    
    #label_position = (top_left[0], top_left[1] - 10)  # Just above the rectangle
    
    # Define the font settings
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #font_scale = 3
    #font_color = (0, 255, 0)
    #font_thickness = 10
    
    # Add the custom label to the image
    #cv2.putText(opencv_image, label_text, label_position, font, font_scale, font_color, font_thickness)
    
    # Draw a red circle on the image
    #center_coordinates = (500, 500)  # Change this to the desired circle center
    #radius = 100
    #color = (0, 0, 255)  # Red color in BGR
    #thickness = 2
    #cv2.circle(opencv_image, center_coordinates, radius, color, thickness)

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


