# Floorplan Analysis System - Installation Guide

## System Requirements

- Python 3.8 or higher
- 8GB RAM (16GB recommended)
- 2GB available disk space
- Supported operating systems: Windows, macOS, Linux

## Installation Steps

### 1. Clone Project
```bash
git clone <repository-url>
cd FYP-Floorplan
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# Or manually install main dependencies
pip install torch torchvision opencv-python numpy pillow matplotlib flask flask-cors
```

### 4. Verify Installation
```bash
# Check if dependencies are correctly installed
python start.py --test

# Run demonstration
python demo.py
```

## Quick Start

### 1. Start System
```bash
python start.py
```

### 2. Access Web Interface
Open browser and visit: http://localhost:5000

### 3. Upload Floorplan
- Click upload area or drag and drop image files
- Set scale ratio (default 1:100)
- Click analyze button

### 4. View Results
- List of detected objects
- Dimension calculation results
- Statistical information

## Troubleshooting

### Common Issues

#### 1. Dependency Installation Failed
```bash
# Upgrade pip
pip install --upgrade pip

# Use domestic mirror source
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 2. Model Files Missing
The system will automatically use simulation detection mode, which does not affect basic functionality demonstration.

#### 3. Port Already in Use
```bash
# Change port
python app.py --port 5001
```

#### 4. Insufficient Memory
- Close other applications
- Use smaller image files
- Adjust detection parameters

### Performance Optimization

#### 1. GPU Acceleration (Optional)
```bash
# Install CUDA version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Memory Optimization
- Use smaller image sizes
- Adjust batch processing size
- Enable memory mapping

## Development Environment Setup

### 1. Code Formatting
```bash
pip install black flake8
black .
flake8 .
```

### 2. Run Tests
```bash
# Run all tests
python run_tests.py

# Run specific tests
python run_tests.py --module models
python run_tests.py --module utils
python run_tests.py --module integration
```

### 3. Debug Mode
```bash
# Enable debug mode
export FLASK_ENV=development
python app.py
```

## Deployment Guide

### 1. Production Environment Configuration
```bash
# Set environment variables
export FLASK_ENV=production
export SECRET_KEY=your-secret-key

# Start application
python app.py
```

### 2. Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

### 3. Cloud Deployment
- Supports Heroku, AWS, Azure and other cloud platforms
- Configure environment variables
- Set up static file services

## Usage Examples

### 1. Basic Usage
```python
from models.floorplan_detector import FloorplanDetector
from utils.image_processor import ImageProcessor
from utils.scale_calculator import ScaleCalculator

# Initialize components
detector = FloorplanDetector()
processor = ImageProcessor()
calculator = ScaleCalculator()

# Process image
image = cv2.imread('floorplan.jpg')
processed = processor.preprocess_array(image)
detections = detector.detect_objects(processed)
calculations = calculator.calculate_sizes(detections, 100)
```

### 2. Batch Processing
```python
import os
from pathlib import Path

# Batch process images in folder
input_dir = Path('input_images')
output_dir = Path('results')

for image_file in input_dir.glob('*.jpg'):
    # Process each image
    image = cv2.imread(str(image_file))
    detections = detector.detect_objects(image)
    calculations = calculator.calculate_sizes(detections, 100)
    
    # Save results
    result_file = output_dir / f"{image_file.stem}_result.json"
    with open(result_file, 'w') as f:
        json.dump(calculations, f, indent=2)
```

### 3. API Usage
```python
import requests

# Upload image for analysis
with open('floorplan.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/upload', files=files)
    
# Get analysis results
data = response.json()
analysis_response = requests.post('http://localhost:5000/analyze', 
                                json={'filename': data['filename'], 'scale_ratio': 100})
```

## Technical Support

### 1. Documentation Resources
- Project documentation: README.md
- API documentation: Check code comments
- Example code: demo.py

### 2. Issue Reporting
- Check log files: logs/app.log
- Run diagnostics: python start.py --test
- View error information: Console output

### 3. Community Support
- GitHub Issues: Report bugs and feature requests
- Technical discussions: Check project discussion area
- Contribute code: Submit Pull Request

## Changelog

### v1.0.0 (2024-10-23)
- ✅ Initial version release
- ✅ Basic functionality implementation
- ✅ Web interface completion
- ✅ Test coverage completion

### Future Plans
- 🔄 Model optimization
- 🔄 Performance improvement
- 🔄 Feature expansion
- 🔄 User experience improvement

---

**After installation, run `python start.py` to start the system!**
