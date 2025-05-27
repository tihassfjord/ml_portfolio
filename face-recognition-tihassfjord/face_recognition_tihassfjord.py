"""
Real-time face recognition system by tihassfjord
Uses OpenCV for video capture and face_recognition for embeddings
"""

import cv2
import face_recognition
import os
import numpy as np
from pathlib import Path

class FaceRecognitionSystem:
    """Real-time face recognition system"""
    
    def __init__(self, faces_dir="faces"):
        self.faces_dir = faces_dir
        self.known_encodings = []
        self.known_names = []
        self.tolerance = 0.6  # Lower = more strict
        
        # Create faces directory if it doesn't exist
        Path(faces_dir).mkdir(exist_ok=True)
        
        # Load known faces
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load and encode known faces from the faces directory"""
        print("Loading known faces...")
        
        if not os.path.exists(self.faces_dir):
            print(f"Faces directory '{self.faces_dir}' not found. Creating it...")
            os.makedirs(self.faces_dir)
            print("Please add face images to the faces/ directory and restart.")
            return
        
        face_files = [f for f in os.listdir(self.faces_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not face_files:
            print("No face images found in faces/ directory.")
            print("Add some .jpg/.png images with person names as filenames.")
            self._create_demo_faces()
            return
        
        for filename in face_files:
            try:
                # Load image
                image_path = os.path.join(self.faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Get face encoding
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    # Use the first face found in the image
                    self.known_encodings.append(encodings[0])
                    # Use filename (without extension) as name
                    name = os.path.splitext(filename)[0]
                    self.known_names.append(name)
                    print(f"  Loaded: {name}")
                else:
                    print(f"  Warning: No face found in {filename}")
                    
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
        
        print(f"Loaded {len(self.known_encodings)} known faces.")
    
    def _create_demo_faces(self):
        """Create demo face images using webcam"""
        print("\nDemo mode: No known faces found.")
        print("The system will detect faces but show 'Unknown' for all faces.")
        print("To add known faces:")
        print("1. Take photos of people you want to recognize")
        print("2. Save them as .jpg files in the faces/ directory")
        print("3. Name files with the person's name (e.g., 'john.jpg')")
        print("4. Restart the application")
    
    def recognize_faces(self, frame):
        """Recognize faces in a frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names = []
        
        for face_encoding in face_encodings:
            # See if the face matches any known faces
            matches = face_recognition.compare_faces(
                self.known_encodings, face_encoding, tolerance=self.tolerance
            )
            name = "Unknown"
            confidence = 0.0
            
            if self.known_encodings:
                # Find the known face with smallest distance
                face_distances = face_recognition.face_distance(
                    self.known_encodings, face_encoding
                )
                
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
            
            face_names.append((name, confidence))
        
        return face_locations, face_names
    
    def draw_results(self, frame, face_locations, face_names):
        """Draw bounding boxes and names on frame"""
        for (top, right, bottom, left), (name, confidence) in zip(face_locations, face_names):
            # Draw rectangle around face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label with name and confidence
            label = f"{name}"
            if name != "Unknown":
                label += f" ({confidence:.2f})"
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw label text
            cv2.putText(frame, label, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        # Add title
        cv2.putText(frame, "tihassfjord Face Recognition", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main loop for real-time face recognition"""
        print("Starting real-time face recognition (tihassfjord).")
        
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open webcam.")
            return
        
        print("Starting webcam feed...")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        
        try:
            while True:
                # Capture frame
                ret, frame = video_capture.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                # Process every other frame for better performance
                if frame_count % 2 == 0:
                    # Resize frame for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    
                    # Recognize faces
                    face_locations, face_names = self.recognize_faces(small_frame)
                    
                    # Scale back up face locations
                    face_locations = [(top*4, right*4, bottom*4, left*4) 
                                    for (top, right, bottom, left) in face_locations]
                
                # Draw results
                frame = self.draw_results(frame, face_locations, face_names)
                
                # Display frame
                cv2.imshow("tihassfjord Face Recognition", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"capture_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame as {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopping face recognition...")
        
        finally:
            # Clean up
            video_capture.release()
            cv2.destroyAllWindows()
            print("Face recognition stopped.")

def main():
    """Main function"""
    try:
        # Create and run face recognition system
        face_system = FaceRecognitionSystem()
        face_system.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam connected and the required packages installed.")
        print("Install with: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
