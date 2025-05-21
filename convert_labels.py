import os
import glob

# Define paths
kitti_label_dir = 'data/labels/training/label_2'
yolo_label_dir = 'data/labels/training/image_2'
image_dir = 'data/images/training/image_2'

# Ensure the output directory exists
os.makedirs(yolo_label_dir, exist_ok=True)

# Get all KITTI label files
kitti_labels = glob.glob(os.path.join(kitti_label_dir, '*.txt'))

# Class mapping - only vehicle classes (0-2)
class_mapping = {
    # Primary vehicle classes
    'Car': 0,
    'Truck': 1,
    # Other vehicles mapped to class 2
    'Van': 2,
    'Tram': 2,
    'Misc': 2,
    'Cyclist': 2,
    # Skip these
    'Pedestrian': None,
    'Person': None,
    'Person_sitting': None,
    'DontCare': None
}

def convert_label_file(kitti_label, yolo_label):
    # Read the KITTI label
    with open(kitti_label, 'r') as f:
        lines = f.readlines()

    # Open the YOLOv5 label file for writing
    with open(yolo_label, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 15:  # Ensure the line has enough parts
                class_name = parts[0]
                
                # Skip if class is not in our mapping or is None
                if class_name not in class_mapping or class_mapping[class_name] is None:
                    continue
                
                # Extract bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(float, parts[4:8])
                
                # Skip if the bounding box is too small (might be noise)
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue
                
                # Calculate center and dimensions
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # Normalize (assuming image dimensions are 1242x375, adjust if different)
                x_center /= 1242
                y_center /= 375
                width /= 1242
                height /= 375
                
                # Get class ID from mapping
                class_id = class_mapping[class_name]
                
                # Write in YOLOv5 format
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# First, convert all KITTI labels to YOLO format
for kitti_label in kitti_labels:
    filename = os.path.basename(kitti_label)
    yolo_label = os.path.join(yolo_label_dir, filename)
    convert_label_file(kitti_label, yolo_label)

# Now fix any remaining class 3 labels
yolo_labels = glob.glob(os.path.join(yolo_label_dir, '*.txt'))
for yolo_label in yolo_labels:
    with open(yolo_label, 'r') as f:
        lines = f.readlines()
    
    # Check if any line starts with '3'
    if any(line.startswith('3 ') for line in lines):
        # Create a new file with corrected labels
        with open(yolo_label, 'w') as f:
            for line in lines:
                if line.startswith('3 '):
                    # Replace class 3 with class 2
                    f.write('2' + line[1:])
                else:
                    f.write(line)

print("Label conversion completed.") 