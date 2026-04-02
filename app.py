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
import csv
from datetime import datetime
import joblib
from model import TrafficLSTM

# Add YOLOv5 to path
sys.path.append('yolov5')

app = Flask(__name__, static_folder='.')
CORS(app)

# Load the model
MODEL_PATH = 'yolov5/runs/train/exp33/weights/best.pt'
FALLBACK_MODEL = 'yolov5s.pt'
model = None
lstm_model = None
scaler = None
HISTORY_FILE = 'traffic_history.csv'

def init_history_file():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'direction', 'car', 'truck', 'bike', 'auto', 'ambulance', 'total'])

def save_to_history(counts, direction='N'):
    init_history_file()
    with open(HISTORY_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            direction,
            counts['car'],
            counts['truck'],
            counts['bike'],
            counts['auto'],
            counts['ambulance'],
            counts['total']
        ])

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

def load_lstm():
    global lstm_model, scaler
    if lstm_model is None:
        try:
            scaler = joblib.load('traffic_scaler.gz')
            lstm_model = TrafficLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)
            lstm_model.load_state_dict(torch.load('traffic_lstm.pt'))
            lstm_model.eval()
            print("LSTM Model and Scaler loaded successfully")
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
    return lstm_model, scaler

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
    
    direction = request.form.get('direction', default='N')
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

        # Create a transparent overlay for bounding boxes
        # results.xyxy[0] has [x1, y1, x2, y2, conf, cls]
        overlay = np.zeros((img.height, img.width, 4), dtype=np.uint8)
        
        # Draw detections on overlay
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            label = f"{results.names[int(cls)]}" # Remove confidence for cleaner look
            color = (0, 255, 100, 200) # Subtle cyan/green with alpha
            
            pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
            
            # Ultra-thin box
            cv2.rectangle(overlay, pt1, pt2, color, 1)
            
            # Elegant small label
            font_scale = 0.35
            tf = 1
            t_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=tf)[0]
            label_y = pt1[1] - 4 if pt1[1] > 15 else pt1[1] + t_size[1] + 4
            # Minimal background for text
            cv2.rectangle(overlay, (pt1[0], label_y - t_size[1] - 2), (pt1[0] + t_size[0], label_y + 2), (0,0,0,150), -1)
            cv2.putText(overlay, label, (pt1[0], label_y), 0, font_scale, (255, 255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

        # Encode transparent overlay as PNG
        _, buffer = cv2.imencode('.png', overlay)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        counts['total'] = sum(counts.values())
        
        # Save to history for LSTM/Analytics
        save_to_history(counts, direction)
        
        return jsonify({
            'type': 'image',
            'counts': counts,
            'image': img_base64
        })
    except Exception as e:
        print(f"Detection Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    limit = request.args.get('limit', default=20, type=int)
    init_history_file()
    history = []
    try:
        with open(HISTORY_FILE, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Return last 'limit' rows
            history = rows[-limit:]
    except Exception as e:
        print(f"History Error: {e}")
    return jsonify(history)

@app.route('/api/predict', methods=['GET'])
def predict():
    direction = request.args.get('direction', default='N')
    loaded_lstm, loaded_scaler = load_lstm()
    
    if loaded_lstm is None:
        return jsonify({'error': 'LSTM Model not loaded'}), 500
        
    try:
        # Get last 24 records for this direction
        init_history_file()
        history = []
        with open(HISTORY_FILE, 'r') as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if row.get('direction') == direction]
            
            if not rows:
                return jsonify({'forecast': [20, 22, 25, 23, 21]})
                
            if len(rows) < 24:
                last_val = int(rows[-1].get('total', 0))
                return jsonify({'forecast': [max(5, last_val + i) for i in range(5)]})
            
            last_24 = rows[-24:]
            counts = []
            for r in last_24:
                try:
                    counts.append(float(r.get('total', 0)))
                except:
                    counts.append(0.0)
            
        # Prepare for LSTM
        inputs = np.array(counts).reshape(-1, 1)
        inputs_scaled = loaded_scaler.transform(inputs)
        inputs_tensor = torch.from_numpy(inputs_scaled).float().unsqueeze(0) # (1, 24, 1)
        
        # Multi-step forecast (next 5 steps)
        forecast = []
        curr_inputs = inputs_tensor
        with torch.no_grad():
            for _ in range(5):
                pred = loaded_lstm(curr_inputs)
                forecast.append(float(loaded_scaler.inverse_transform(pred.numpy())[0][0]))
                # Slide window
                curr_inputs = torch.cat((curr_inputs[:, 1:, :], pred.unsqueeze(0)), dim=1)
        
        return jsonify({'forecast': [int(f) for f in forecast]})
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    load_lstm()
    app.run(host='0.0.0.0', port=8000, debug=True)
