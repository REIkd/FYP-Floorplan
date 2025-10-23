# Floorplan Analysis System

A deep learning-based floorplan recognition and analysis system that automatically identifies objects in floorplans and calculates dimensions.

## Features

- 🏗️ **Object Recognition**: Automatically identify architectural elements like doors, windows, stairs, elevators in floorplans
- 📏 **Dimension Calculation**: Calculate room areas and object dimensions based on scale ratios
- 📊 **Statistical Analysis**: Count and analyze the distribution of various objects
- 🖥️ **User Interface**: User-friendly web interface with image upload and analysis
- 🔍 **Data Validation**: Complete dataset validation and testing functionality

## Project Structure

```
FYP-Floorplan/
├── app/                    # Flask Web Application
├── models/                 # Deep Learning Models
├── data/                   # Dataset and Preprocessing
├── utils/                  # Utility Functions
├── tests/                  # Test Files
└── requirements.txt        # Dependencies
```

## Installation and Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Access http://localhost:5000 to use the web interface

## Dataset Requirements

- Floorplan image formats: JPG, PNG
- Include scale information
- Annotation files: YOLO format
- Object categories: doors, windows, stairs, elevators, rooms, etc.

## Technology Stack

- **Deep Learning**: PyTorch, YOLOv8
- **Image Processing**: OpenCV, PIL
- **Web Framework**: Flask
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
