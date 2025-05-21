# Vehicle Detection System

A real-time vehicle detection system using YOLOv5 and Streamlit, capable of detecting and classifying vehicles in images with high accuracy.

## Model Choice and Approach

### Why YOLOv5?
- **Speed**: Real-time detection capabilities
- **Accuracy**: State-of-the-art object detection
- **Efficiency**: Lightweight model (YOLOv5n) for faster processing
- **Flexibility**: Easy to train and fine-tune

### Model Architecture
- Base Model: YOLOv5n (nano)
- Input Size: 480x480 pixels
- Classes: 
  - Car (primary)
  - Truck (primary)
  - Other vehicles (secondary)
- Training Parameters:
  - Batch Size: 24
  - Epochs: 30
  - Learning Rate: Adaptive
  - Optimizer: SGD with momentum

### Detection Approach
1. Image Preprocessing:
   - Resize to 480x480
   - Normalize pixel values
   - Convert to tensor format

2. Feature Extraction:
   - Convolutional layers
   - Feature pyramid network
   - Multi-scale detection

3. Post-processing:
   - Non-maximum suppression
   - Confidence thresholding
   - Bounding box refinement

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Git

### Installation Steps

1. Clone the repository:
```bash
git clone [repository-url]
cd vehicle-detection-system
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLOv5:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cd ..
```

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to:
```
http://localhost:8501
```

3. Upload an image and adjust confidence threshold as needed

## Limitations and Assumptions

### Technical Limitations
1. **Performance**:
   - CPU-only mode is slower
   - Minimum 4GB RAM required
   - GPU recommended for real-time detection

2. **Detection Accuracy**:
   - Works best with clear, well-lit images
   - May struggle with:
     - Extreme weather conditions
     - Poor lighting
     - Occluded vehicles
     - Very small vehicles

3. **Image Requirements**:
   - Minimum resolution: 480x480 pixels
   - Supported formats: JPG, JPEG, PNG
   - Maximum file size: 10MB

### Assumptions
1. **Vehicle Types**:
   - Primary focus on cars and trucks
   - Other vehicles detected but not specifically classified
   - Assumes vehicles are reasonably visible

2. **Environment**:
   - Standard road/street scenes
   - Moderate to good lighting conditions
   - Vehicles are not heavily occluded

3. **Usage**:
   - Single image processing
   - Static images (not video)
   - User can adjust confidence threshold

## Observations and Notes

### Performance Metrics
- Average detection speed: 100-200ms per image
- Typical confidence threshold: 0.25-0.5
- Best results with:
  - Clear images
  - Good lighting
  - Unobstructed views

### Known Issues
1. **Detection**:
   - May miss very small vehicles
   - Can confuse similar vehicle types
   - Performance drops in poor lighting

2. **Classification**:
   - Limited to three main categories
   - May misclassify unusual vehicle types
   - Confidence scores may vary

### Future Improvements
1. **Model Enhancements**:
   - Add more vehicle classes
   - Improve small object detection
   - Better handling of occlusions

2. **Features**:
   - Video processing support
   - Real-time tracking
   - Multiple model options

## Project Structure
```
vehicle-detection-system/
├── app.py                 # Streamlit application
├── data/
│   ├── dataset.yaml      # Dataset configuration
│   ├── images/           # Training and test images
│   └── labels/           # Training labels
├── yolov5/               # YOLOv5 framework
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Sharing the Project

### For Repository Owners

1. **Prepare Your Repository**:
   ```bash
   # Initialize git repository (if not already done)
   git init
   
   # Add all files
   git add .
   
   # Create initial commit
   git commit -m "Initial commit: Vehicle Detection System"
   ```

2. **Create a GitHub Repository**:
   - Go to GitHub.com and sign in
   - Click "New repository"
   - Name it "vehicle-detection-system"
   - Don't initialize with README (we already have one)
   - Click "Create repository"

3. **Link and Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/vehicle-detection-system.git
   git branch -M main
   git push -u origin main
   ```

4. **Share the Repository**:
   - Share the repository URL: `https://github.com/YOUR_USERNAME/vehicle-detection-system`
   - Add collaborators in repository settings if needed
   - Create issues for known bugs or feature requests

### For Users/Contributors

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/vehicle-detection-system.git
   cd vehicle-detection-system
   ```

2. **Follow Setup Instructions**:
   - Create virtual environment
   - Install dependencies
   - Download YOLOv5
   - Follow the setup steps in the README

3. **Running the Project**:
   ```bash
   streamlit run app.py
   ```

### What to Include When Sharing

1. **Essential Files**:
   - `app.py` (main application)
   - `requirements.txt` (dependencies)
   - `README.md` (documentation)
   - `data/dataset.yaml` (dataset configuration)
   - `.gitignore` (git ignore rules)

2. **Optional but Recommended**:
   - Sample test images
   - Pre-trained weights (if allowed by license)
   - Example configuration files
   - Documentation for custom modifications

### What NOT to Include

1. **Exclude These Files**:
   - Virtual environment folder (`venv/`)
   - Large model weights (share download links instead)
   - Personal API keys or credentials
   - Large datasets (share download instructions)
   - Cache files and temporary data
   - IDE-specific files

### Best Practices for Sharing

1. **Documentation**:
   - Keep README.md up to date
   - Document any changes or customizations
   - Include troubleshooting guides
   - Add comments in code

2. **Version Control**:
   - Use meaningful commit messages
   - Create branches for new features
   - Tag releases for stable versions
   - Keep repository clean and organized

3. **Collaboration**:
   - Use issues for bug reports
   - Create pull requests for contributions
   - Respond to issues and PRs promptly
   - Maintain a code of conduct

4. **Updates and Maintenance**:
   - Regular dependency updates
   - Security patches
   - Performance improvements
   - New feature additions

### Getting Help

If users need help:
1. Check the README.md first
2. Look for existing issues
3. Create a new issue if needed
4. Contact maintainers through GitHub 