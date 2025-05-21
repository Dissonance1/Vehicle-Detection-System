import subprocess
import os
import sys

def run_training():
    # Training command with all parameters
    cmd = """
    python yolov5/train.py \
        --img 480 \
        --batch 24 \
        --epochs 30 \
        --data data/dataset.yaml \
        --weights yolov5/yolov5n.pt \
        --workers 4 \
        --cache
    """
    
    print("Starting YOLOv5 training with parameters:")
    print("Image size: 480x480")
    print("Batch size: 24")
    print("Epochs: 30")
    print("Dataset: data/dataset.yaml")
    print("Weights: yolov5/yolov5n.pt")
    print("Workers: 4")
    print("Cache: Enabled")
    print("\nTraining started...")
    
    try:
        # Run the training command
        subprocess.run(cmd, shell=True, check=True)
        print("\nTraining completed successfully!")
        print("Results saved in yolov5/runs/train/")
    except subprocess.CalledProcessError as e:
        print(f"\nError during training: {e}")
        return
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return

if __name__ == "__main__":
    run_training() 