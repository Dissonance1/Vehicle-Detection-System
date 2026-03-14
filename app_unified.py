import os
import sys
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import io
import base64
import tempfile

# Add YOLOv5 to path
sys.path.append('yolov5')

app = Flask(__name__, static_folder='.')
CORS(app)

# Load the model
MODEL_PATH = 'yolov5/runs/train/exp33/weights/best.pt'
FALLBACK_MODEL = 'yolov5s.pt'
model = None

def load_model():
    global model
    if model is None:
        try:
            # Explicitly check for CUDA and handle potential mismatches
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Initializing YOLOv5 on {device}...")
            
            # Use torch hub to load the model
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device=device)
            print(f"Model loaded successfully on {device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to CPU if CUDA fails
            try:
                print("Attempting CPU fallback...")
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device='cpu')
                print("Model loaded successfully on CPU")
            except Exception as e2:
                print(f"Final model loading error: {e2}")
    return model

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('.', path)

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    filename = file.filename.lower()
    
    loaded_model = load_model()
    if loaded_model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Standardize: Treat everything as an image frame for real-time performance
        img_bytes = file.read()
        # Handle OpenCV frame or PIL image
        img = Image.open(io.BytesIO(img_bytes))
        
        # Inference (GPU accelerated if available)
        results = loaded_model(img)
        
        # Process counts
        counts = {'car': 0, 'truck': 0, 'bike': 0, 'auto': 0, 'ambulance': 0}
        
        # YOLOv5 COCO indices: 2:car, 3:motorcycle, 5:bus, 7:truck
        for det in results.xyxy[0]:
            class_id = int(det[5])
            if class_id == 2: counts['car'] += 1
            elif class_id == 7 or class_id == 5: counts['truck'] += 1 # Grouping large vehicles
            elif class_id == 3 or class_id == 1: counts['bike'] += 1
            elif class_id == 0: pass # Ignoring person for traffic stats
            
        # Custom fallback for binary models (if any)
        if max(counts.values()) == 0:
            for det in results.xyxy[0]:
                if int(det[5]) == 0: counts['car'] += 1
                elif int(det[5]) == 1: counts['truck'] += 1

        # Render annotated frame
        rendered_img = results.render()[0]
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        counts['total'] = sum(counts.values())
        return jsonify({
            'type': 'image',
            'counts': counts,
            'image': img_base64
        })
    except Exception as e:
        print(f"Detection Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8000, debug=True)
