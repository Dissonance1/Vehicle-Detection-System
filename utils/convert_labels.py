import os
from pathlib import Path
import cv2
import shutil

def convert_kitti_to_yolo(kitti_label_path, yolo_label_path, img_width, img_height):
    """Convert KITTI format labels to YOLO format"""
    try:
        with open(kitti_label_path, 'r') as f:
            lines = f.readlines()
        
        yolo_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 15:  # KITTI format has at least 15 values
                continue
                
            # KITTI format: class_name truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom height width length x y z rotation_y
            class_name = parts[0]
            
            # Convert class name to YOLO format (0 for car, 1 for truck)
            if class_name.lower() in ['car', 'truck']:
                class_id = 0 if class_name.lower() == 'car' else 1
                
                # Get bounding box coordinates
                bbox_left = float(parts[4])
                bbox_top = float(parts[5])
                bbox_right = float(parts[6])
                bbox_bottom = float(parts[7])
                
                # Convert to YOLO format (x_center, y_center, width, height)
                x_center = (bbox_left + bbox_right) / 2 / img_width
                y_center = (bbox_top + bbox_bottom) / 2 / img_height
                width = (bbox_right - bbox_left) / img_width
                height = (bbox_bottom - bbox_top) / img_height
                
                # Add to YOLO format lines
                yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        # Write YOLO format labels
        with open(yolo_label_path, 'w') as f:
            f.writelines(yolo_lines)
        return True
    except Exception as e:
        print(f"Error converting {kitti_label_path}: {str(e)}")
        return False

def main():
    # Create label directories if they don't exist
    os.makedirs('data/labels/train', exist_ok=True)
    os.makedirs('data/labels/val', exist_ok=True)
    
    # Find the first image to get dimensions
    img_path = None
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        img_path = next(Path('data/images/train').glob(ext), None)
        if img_path:
            break
    
    if not img_path:
        print("Error: No images found in data/images/train")
        return
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return
    
    img_height, img_width = img.shape[:2]
    print(f"Image dimensions: {img_width}x{img_height}")
    
    # Find KITTI labels
    kitti_label_dirs = [
        'data/labels/training/label_2',
        'data/labels/training',
        'data/labels/train',
        'data/labels'
    ]
    
    kitti_train_labels = None
    for dir_path in kitti_label_dirs:
        if os.path.exists(dir_path):
            kitti_train_labels = Path(dir_path)
            print(f"Found KITTI labels in: {dir_path}")
            break
    
    if not kitti_train_labels:
        print("Error: Could not find KITTI labels")
        return
    
    # Convert training labels
    success_count = 0
    total_count = 0
    
    for label_file in kitti_train_labels.glob('*.txt'):
        total_count += 1
        yolo_label_path = f'data/labels/train/{label_file.name}'
        if convert_kitti_to_yolo(str(label_file), yolo_label_path, img_width, img_height):
            success_count += 1
    
    print(f"\nLabel conversion complete!")
    print(f"Successfully converted {success_count} out of {total_count} labels")
    
    # Copy some labels to validation set
    if success_count > 0:
        val_count = min(success_count // 5, 100)  # Use 20% of labels for validation
        train_labels = list(Path('data/labels/train').glob('*.txt'))
        for label_file in train_labels[:val_count]:
            shutil.copy2(label_file, f'data/labels/val/{label_file.name}')
        print(f"Created validation set with {val_count} labels")

if __name__ == '__main__':
    main() 
