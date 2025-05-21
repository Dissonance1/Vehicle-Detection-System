import os
import shutil
from pathlib import Path
import random
import cv2
import numpy as np

def create_directory_structure():
    """Create the necessary directory structure for the dataset"""
    directories = [
        'data/images/train',
        'data/images/val',
        'data/labels/train',
        'data/labels/val'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def split_dataset(source_dir, train_ratio=0.8):
    """Split dataset into training and validation sets"""
    source_dir = Path(source_dir)
    image_files = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Copy images and labels
    for img_path in train_files:
        # Copy image
        shutil.copy2(img_path, 'data/images/train')
        # Copy corresponding label if exists
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            shutil.copy2(label_path, 'data/labels/train')
    
    for img_path in val_files:
        # Copy image
        shutil.copy2(img_path, 'data/images/val')
        # Copy corresponding label if exists
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            shutil.copy2(label_path, 'data/labels/val')

def validate_labels():
    """Validate label files and convert to YOLO format if needed"""
    for split in ['train', 'val']:
        label_dir = Path(f'data/labels/{split}')
        for label_file in label_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # Convert labels if needed (assuming COCO format)
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:  # YOLO format
                    new_lines.append(line)
                else:  # Convert from COCO format
                    # Add conversion logic here if needed
                    pass
            
            # Write back converted labels
            with open(label_file, 'w') as f:
                f.writelines(new_lines)

def visualize_dataset(split='train', num_samples=5):
    """Visualize random samples from the dataset with bounding boxes"""
    image_dir = Path(f'data/images/{split}')
    label_dir = Path(f'data/labels/{split}')
    
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    for img_path in samples:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Read labels
        label_path = label_dir / img_path.with_suffix('.txt').name
        if not label_path.exists():
            continue
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Draw bounding boxes
        h, w = img.shape[:2]
        for line in lines:
            cls, x_center, y_center, width, height = map(float, line.strip().split())
            
            # Convert normalized coordinates to pixel coordinates
            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            class_name = 'car' if cls == 0 else 'truck'
            cv2.putText(img, class_name, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization
        output_dir = Path('data/visualizations')
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_dir / f'{split}_{img_path.name}'), img)

if __name__ == '__main__':
    # Create directory structure
    create_directory_structure()
    
    # Ask for source directory
    source_dir = input("Enter the path to your dataset directory: ")
    if os.path.exists(source_dir):
        # Split dataset
        split_dataset(source_dir)
        
        # Validate labels
        validate_labels()
        
        # Visualize samples
        print("\nVisualizing training samples...")
        visualize_dataset('train')
        print("\nVisualizing validation samples...")
        visualize_dataset('val')
        
        print("\nDataset preparation complete!")
        print("Check the 'data/visualizations' directory for sample visualizations.")
    else:
        print(f"Error: Directory {source_dir} does not exist!") 