import os
from pathlib import Path
import cv2
import shutil
import yaml

def convert_labels(original_label_path, yolo_label_path, img_width, img_height):
    """Convert labels from original format to YOLO format with specified class mapping."""
    try:
        with open(original_label_path, 'r') as f:
            lines = f.readlines()
        
        yolo_lines = []
        for line in lines:
            parts = line.strip().split()
            # Ensure the line has enough parts for a bounding box (assuming YOLO-like structure: class_id x y w h)
            if len(parts) < 5: 
                continue
                
            # Assuming the first part is the original class ID
            try:
                original_class_id = int(parts[0])
            except ValueError:
                # Skip line if class ID is not an integer
                continue
            
            # Define a dictionary for original class ID to new YOLO class ID mapping
            # Mapping: Original ID -> New ID (0 for Car, 1 for Truck, etc.)
            class_id_mapping = {
                5: 0,   # Car
                18: 1,  # Truck
                4: 2,   # Bus
                10: 4,  # Motorbike
                13: 5   # Bicycle
                # If you have an original ID for Ambulance and want to include it (New ID 3), add it here.
            }

            # Check if the original class ID is in our mapping
            if original_class_id in class_id_mapping:
                new_class_id = class_id_mapping[original_class_id]
                
                # Get bounding box coordinates (parts[1] to parts[4] for YOLO format already)
                # These are assumed to be already normalized x_center, y_center, width, height
                try:
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except (ValueError, IndexError):
                     # Skip line if bounding box coordinates are not valid numbers
                    continue

                
                # Add to YOLO format lines
                yolo_lines.append(f"{new_class_id} {x_center} {y_center} {width} {height}\n")

        # Write YOLO format labels
        with open(yolo_label_path, 'w') as f:
            f.writelines(yolo_lines)
        return True
    except Exception as e:
        print(f"Error converting {original_label_path}: {str(e)}")
        return False

def main():
    # Read dataset configuration
    dataset_config_path = 'data/dataset.yaml'
    if not os.path.exists(dataset_config_path):
        print(f"Error: Dataset configuration file not found at {dataset_config_path}")
        return
        
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    dataset_path = dataset_config.get('path')
    train_images_rel_path = dataset_config.get('train')
    val_images_rel_path = dataset_config.get('val')

    if not dataset_path or not train_images_rel_path or not val_images_rel_path:
        print("Error: Missing path, train, or val in dataset.yaml")
        return
        
    train_images_path = Path(dataset_path) / train_images_rel_path
    val_images_path = Path(dataset_path) / val_images_rel_path

    # Create output label directories if they don't exist
    os.makedirs('data/labels/train', exist_ok=True)
    os.makedirs('data/labels/val', exist_ok=True)
    
    # Find the first image in the training set to get dimensions
    img_path = None
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        img_path = next(train_images_path.glob(ext), None)
        if img_path:
            break
    
    if not img_path:
        print(f"Error: No images found in {train_images_path}")
        return
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return
    
    img_height, img_width = img.shape[:2]
    print(f"Image dimensions: {img_width}x{img_height}")
    
    # Assuming labels are in a 'labels' directory parallel to 'images' in your original data path
    train_original_labels_path = Path(dataset_path) / train_images_rel_path.replace('images', 'labels')
    val_original_labels_path = Path(dataset_path) / val_images_rel_path.replace('images', 'labels')

    if not train_original_labels_path.exists():
        print(f"Error: Training labels directory not found at {train_original_labels_path}")
        return

    # Convert training labels
    success_count = 0
    total_count = 0
    
    for label_file in train_original_labels_path.glob('*.txt'):
        total_count += 1
        # Define the output path for the converted label
        yolo_label_path = f'data/labels/train/{label_file.name}'
        if convert_labels(str(label_file), yolo_label_path, img_width, img_height):
            success_count += 1
    
    print(f"\nTraining label conversion complete!")
    print(f"Successfully converted {success_count} out of {total_count} labels")

    # Convert validation labels
    success_count_val = 0
    total_count_val = 0

    if val_original_labels_path.exists():
        for label_file in val_original_labels_path.glob('*.txt'):
            total_count_val += 1
            # Define the output path for the converted label
            yolo_label_path = f'data/labels/val/{label_file.name}'
            if convert_labels(str(label_file), yolo_label_path, img_width, img_height):
                success_count_val += 1

        print(f"\nValidation label conversion complete!")
        print(f"Successfully converted {success_count_val} out of {total_count_val} labels")
    else:
        print(f"Warning: Validation labels directory not found at {val_original_labels_path}. Skipping validation label conversion.")


if __name__ == '__main__':
    main() 
