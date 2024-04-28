import cv2
import numpy as np

def extract_faces(video_path, output_size=(224, 224)):
    # Load the pre-trained Haar Cascade model for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return None
    
    faces = []  # List to store cropped face images
    frame_count = 0  # Frame counter
    
    while True:
        ret, frame = cap.read()  # Read a frame
        if not ret:
            break  # Break the loop if there are no frames left

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        detections = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        # Crop faces and append to the list
        for (x, y, w, h) in detections:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, output_size)  # Resize face to desired size
            faces.append(face)
        
        # Increment frame count
        frame_count += 1

    cap.release()  # Release the video capture object
    
    return faces

""" # Usage
video_path = 'path_to_your_video.mp4'
faces = extract_faces(video_path)

# Optionally, display the first few faces
for face in faces[:5]:
    cv2.imshow('Face', face)
    cv2.waitKey(0)  # Wait for a key press to show the next face
cv2.destroyAllWindows()
 """