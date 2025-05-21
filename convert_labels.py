import os
import yaml
import cv2
import numpy as np

def convert_labels():
    # Read dataset paths from dataset.yaml
    with open('data/dataset.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Get image paths for dimensions
    train_img_dir = os.path.join(dataset_config['path'], 'train/images')
    val_img_dir = os.path.join(dataset_config['path'], 'valid/images')
    
    # Original class IDs from the label files:
    # bus =4 truck =18 car =5  bicycle =13 motorbike =10
    class_id_mapping = {
        5: 0,   # Car -> 0
        18: 1,  # Truck -> 1
        4: 2,   # Bus -> 2
        10: 4,  # Motorbike -> 4
        13: 5   # Bicycle -> 5
    }
    
    # Create output directories if they don't exist
    os.makedirs('data/labels/train', exist_ok=True)
    os.makedirs('data/labels/val', exist_ok=True)
    
    # Process training labels
    train_labels_dir = os.path.join(dataset_config['path'], 'train/labels')
    train_count = 0
    for label_file in os.listdir(train_labels_dir):
        if label_file.endswith('.txt'):
            input_path = os.path.join(train_labels_dir, label_file)
            output_path = os.path.join('data/labels/train', label_file)
            
            # Get image dimensions - This part might not be necessary if labels are already normalized
            # img_file = label_file.replace('.txt', '.jpg')
            # img_path = os.path.join(train_img_dir, img_file)
            # if not os.path.exists(img_path):
            #     print(f"Warning: Image {img_file} not found, skipping label {label_file}")
            #     continue
            #     
            # img = cv2.imread(img_path)
            # if img is None:
            #     print(f"Warning: Could not read image {img_file}, skipping label {label_file}")
            #     continue
            #     
            # img_height, img_width = img.shape[:2]
            
            # Read and convert labels
            with open(input_path, 'r') as f:
                lines = f.readlines()
            
            converted_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    if class_id in class_id_mapping:
                        # Convert class ID
                        new_class_id = class_id_mapping[class_id]
                        # Keep the rest of the values (x_center, y_center, width, height) as they are
                        converted_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                        converted_lines.append(converted_line)
            
            # Write converted labels
            with open(output_path, 'w') as f:
                f.writelines(converted_lines)
            
            train_count += 1
    
    # Process validation labels
    val_labels_dir = os.path.join(dataset_config['path'], 'valid/labels')
    val_count = 0
    for label_file in os.listdir(val_labels_dir):
        if label_file.endswith('.txt'):
            input_path = os.path.join(val_labels_dir, label_file)
            output_path = os.path.join('data/labels/val', label_file)
            
            # Get image dimensions - This part might not be necessary if labels are already normalized
            # img_file = label_file.replace('.txt', '.jpg')
            # img_path = os.path.join(val_img_dir, img_file)
            # if not os.path.exists(img_path):
            #     print(f"Warning: Image {img_file} not found, skipping label {label_file}")
            #     continue
            #     
            # img = cv2.imread(img_path)
            # if img is None:
            #     print(f"Warning: Could not read image {img_file}, skipping label {label_file}")
            #     continue
            #     
            # img_height, img_width = img.shape[:2]
            
            # Read and convert labels
            with open(input_path, 'r') as f:
                lines = f.readlines()
            
            converted_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    if class_id in class_id_mapping:
                        # Convert class ID
                        new_class_id = class_id_mapping[class_id]
                        # Keep the rest of the values (x_center, y_center, width, height) as they are
                        converted_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                        converted_lines.append(converted_line)
            
            # Write converted labels
            with open(output_path, 'w') as f:
                f.writelines(converted_lines)
            
            val_count += 1
    
    print(f"Successfully converted {train_count} training labels and {val_count} validation labels")

if __name__ == "__main__":
    convert_labels() 