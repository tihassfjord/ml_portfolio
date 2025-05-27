# Real-time Face Recognition System (highlight) — tihassfjord

## Goal
Detect and recognize faces in real time from webcam feed using OpenCV and deep learning embeddings.

## Dataset
- Custom face images in `faces/` directory
- Real-time webcam feed for testing

## Requirements
- Python 3.8+
- opencv-python
- face_recognition
- numpy
- Pillow

## Setup
1. Create a `faces/` directory
2. Add known face images (JPG format) with filenames as person names
3. Run the face recognition system

## How to Run
```bash
python face_recognition_tihassfjord.py
```

Press 'q' to quit the application.

## Example Output
```
Starting real-time face recognition (tihassfjord).
Loading known faces...
Loaded 3 known faces.
Starting webcam feed...
[Display window with bounding boxes and names, or "Unknown" if not matched]
```

## Project Structure
```
face-recognition-tihassfjord/
│
├── face_recognition_tihassfjord.py    # Main recognition system
├── faces/                             # Directory for known faces
│   ├── person1.jpg                   # Example face images
│   └── person2.jpg
├── requirements.txt                   # Dependencies
└── README.md                         # This file
```

## Key Features
- Real-time face detection and recognition
- Support for multiple known faces
- Live webcam processing
- Bounding box visualization
- High accuracy face embeddings
- Easy to add new faces

## Learning Outcomes
- Computer vision and face recognition
- Real-time video processing
- OpenCV and face_recognition library usage
- Live camera integration
- Performance optimization for real-time systems

## Note
This project requires a webcam. If no known faces are found in the `faces/` directory, the system will still run but only detect faces without recognition.

---
*Project by tihassfjord - Advanced ML Portfolio*
