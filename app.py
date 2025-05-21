import streamlit as st
import torch
from PIL import Image
import io
import os
import sys

# Add YOLOv5 to path
sys.path.append('yolov5')

# Load the model
@st.cache_resource
def load_model():
    model_path = 'yolov5/runs/train/exp33/weights/best.pt'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please complete the training process first.")
        st.info("The model file should be located at: " + model_path)
        return None
    
    try:
        model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title("Vehicle Detection System")
    st.write("Upload an image to detect vehicles")

    # Add confidence threshold slider
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
                             help="Adjust the minimum confidence threshold for detections")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Add a button to trigger detection
        if st.button("Detect Vehicles"):
            # Load model
            model = load_model()
            
            if model is not None:
                try:
                    # Set confidence threshold
                    model.conf = conf_threshold
                    
                    # Perform detection
                    results = model(image)
                    
                    # Display results
                    st.image(results.render()[0], caption="Detection Results", use_column_width=True)
                    
                    # Display detection information
                    st.write("Detection Results:")
                    
                    # Count detections by class
                    car_count = 0
                    truck_count = 0
                    # Remove other_vehicle_count as we only have car and truck classes now
                    # other_vehicle_count = 0

                    for det in results.xyxy[0]:  # xyxy format
                        x1, y1, x2, y2, conf, cls = det.tolist()
                        class_id = int(cls)
                        # Ensure class_id is 0 or 1 based on our 2 classes
                        if class_id == 0:
                            class_name = "Car"
                            car_count += 1
                            st.write(f"Detected {class_name} with confidence: {conf:.2f}")
                        elif class_id == 1:
                            class_name = "Truck"
                            truck_count += 1
                            st.write(f"Detected {class_name} with confidence: {conf:.2f}")
                        # Removed the else block for 'Other Vehicle'
                        # else:
                        #     class_name = "Other Vehicle"
                        #     other_vehicle_count += 1
                        #     st.write(f"Detected {class_name} with confidence: {conf:.2f}")

                    # Display summary
                    st.write("Summary:")
                    st.write(f"Total Cars: {car_count}")
                    st.write(f"Total Trucks: {truck_count}")
                    # Removed the display of other_vehicle_count
                    # st.write(f"Total Other Vehicles: {other_vehicle_count}")
                    # Adjusted the total count to only include car and truck
                    st.write(f"Total Vehicles: {car_count + truck_count}")
                    
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")

if __name__ == "__main__":
    main() 