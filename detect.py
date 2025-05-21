import os
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--weights', type=str, default='runs/train/exp/weights/best.pt', help='Path to model weights')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    return parser.parse_args()

def process_image(img_path, model, conf_thres, iou_thres):
    """Process a single image and return detections"""
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error reading image: {img_path}")
        return None
    
    # Run inference
    results = model(img)
    
    # Process results
    detections = []
    for pred in results.pred[0]:
        x1, y1, x2, y2, conf, cls = pred.cpu().numpy()
        if conf >= conf_thres:
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class': int(cls)
            })
    
    return img, detections

def draw_detections(img, detections, class_names):
    """Draw bounding boxes and labels on image"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        cls = det['class']
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_names[cls]} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.weights)
    model.conf = args.conf_thres
    model.iou = args.iou_thres
    
    # Class names
    class_names = ['car', 'truck']
    
    # Process input
    source = Path(args.source)
    if source.is_file():
        # Single image
        img, detections = process_image(source, model, args.conf_thres, args.iou_thres)
        if img is not None:
            img = draw_detections(img, detections, class_names)
            output_path = os.path.join(args.output, source.name)
            cv2.imwrite(output_path, img)
            print(f"Saved result to {output_path}")
    else:
        # Directory of images
        for img_path in source.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img, detections = process_image(img_path, model, args.conf_thres, args.iou_thres)
                if img is not None:
                    img = draw_detections(img, detections, class_names)
                    output_path = os.path.join(args.output, img_path.name)
                    cv2.imwrite(output_path, img)
                    print(f"Saved result to {output_path}")

if __name__ == '__main__':
    main() 