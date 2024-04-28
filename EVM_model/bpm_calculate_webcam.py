# numerical operations
import numpy as np
# computer vision
import cv2
# system-specific parameters and functions
import sys
import cvzone
# FaceDetector from cvzone.FaceDetectionModule for face detection.
from cvzone.FaceDetectionModule import FaceDetector
# LivePlot from cvzone.PlotModule for live plotting
from cvzone.PlotModule import LivePlot
#  for time-related functions
import time
# for reading and writing CSV files
import csv

# Constants for video dimensions and parameters
realWidth = 640   # real video frame width dimensions
realHeight = 480  # real video frame height dimensions
videoWidth = 160  # proccessed video frame width dimensions
videoHeight = 120  # proccessed video frame height dimensions
videoChannels = 3    # number of video channels in the video frame
videoFrameRate = 15  # frame rate of video

# Initialize face detector and BPM values list
detector = FaceDetector()
# List to store BPM values
bpm_values = []  

# Color Magnification Parameters
levels = 3  # Number of levels in the Gaussian pyramid for face image processing
alpha = 170 # Amplification factor for enhancing heart rate signals
minFrequency = 1.0 # Minimum frequencies for the bandpass filter in heart rate analysis
maxFrequency = 2.0 # Maximum frequencies for the bandpass filter in heart rate analysis
bufferSize = 150 # Size of the buffer for storing processed frames
bufferIndex = 0  # Index for tracking the current position in the buffer

# initialize live plot for BPM visualization
plotY = LivePlot(realWidth,realHeight,[60,120],invert=True)

# Helper Methods
#construct a Gaussian pyramid for the given frame
# Function to construct a Gaussian pyramid for the given frame.
# Parameters:
#   - frame: The input image frame for which the Gaussian pyramid is to be constructed.
#   - levels: The number of levels in the Gaussian pyramid.
# Returns:
#   - pyramid: A list containing the levels of the Gaussian pyramid, where each level is a downsampled version of the input frame.
def buildGauss(frame, levels):
    pyramid = [frame] #Initialize the pyramid list with the original frame at the top level.
    for level in range(levels): # Iterate through each level of the pyramid.
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid
# Function to reconstruct a frame from the Gaussian pyramid.
# Parameters:
#   - pyramid: The list containing the levels of the Gaussian pyramid.
#   - index: The index of the level in the pyramid from which the frame is to be reconstructed.
#   - levels: The number of levels in the Gaussian pyramid.
# Returns:
#   - filteredFrame: The reconstructed frame is obtained from the Gaussian pyramid.
def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels): # Iterate through each level of the pyramid
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame


# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (30, 40) # Location coordinates for displaying loading text on the video frame.
bpmTextLocation = (videoWidth//2, 40) # Location coordinates for displaying BPM (Beats Per Minute) text on the video frame.
fpsTextLoaction = (500,600) # Location coordinates for displaying FPS (Frames Per Second) text on the video frame.

# initialize graph items
fontScale = 1
fontColor = (255,255,255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels)) # Initialize the video Gaussian buffer to store Gaussian pyramid levels for multiple frames.
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency) # Create a mask to specify frequencies within the desired range for the bandpass filter.

# Heart Rate Calculation Variables
bpmCalculationFrequency = 10   #15
bpmBufferIndex = 0 # Index to track the position in the BPM (Beats Per Minute) buffer.
bpmBufferSize = 10 # Size of the BPM buffer.
bpmBuffer = np.zeros((bpmBufferSize)) # Array to store BPM values.

# variables for timing
i = 0
ptime = 0
ftime = 0

# REPLACE WITH YOUR VIDEO PATH !!!!!
video_path = "C:/Users/vishn/OneDrive/Desktop/DATASETS!!!/With_HR_precise_sync/Driver 4/session_1636641409569/video_1636641427815_88398/video_1636641427815_88398.mp4"

# open video filr
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print("Error: Could not open video file.")
    sys.exit()
    
# Main loop for processing video frames
while (True):
    # Read a frame from the video
    ret, frame = video.read()
    if ret == False:
        break
    # Resize frame to desired dimensions
    frame = cv2.resize(frame, (realWidth, realHeight))
       
    # Detect faces in the frame
    frame, bboxs = detector.findFaces(frame,draw=False)
    frameDraw = frame.copy()

    # Calculate frames per second (FPS)
    ftime = time.time()
    fps = 1 / (ftime - ptime)
    ptime = ftime
    # Display FPS on the frame
    cv2.putText(frameDraw, f'FPS: {int(fps)}', (30, 440), 0, 1, [0, 0, 0], thickness=2, lineType=cv2.LINE_AA)
    # If faces are detected
    if bboxs:
        # Extract bounding box coordinates of the first detected face.
        x1, y1, w1, h1 = bboxs[0]['bbox']
        # Draw a rectangle around the detected face on the frame.
        cv2.rectangle(frameDraw, bboxs[0]['bbox'], (255, 0, 255), 2)
        # Extract the region of interest (ROI) containing the detected face.
        detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
        # Resize the detected face to match the desired video dimensions.
        detectionFrame = cv2.resize(detectionFrame,(videoWidth,videoHeight))

        # Construct Gaussian Pyramid for the detected face
        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
        # Compute the Fourier transform of the Gaussian pyramid along the specified axis.
        fourierTransform = np.fft.fft(videoGauss, axis=0)

        # Apply bandpass filter to the Fourier transform
        fourierTransform[mask == False] = 0

        # Extract heart rate information
        if bufferIndex % bpmCalculationFrequency == 0:
            i = i + 1
            # Compute the average of the real part of the Fourier transform for each frame in the buffer.
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            # Find the dominant frequency in the Fourier transform average.
            hz = frequencies[np.argmax(fourierTransformAvg)]
            # Calculate the heart rate in beats per minute (BPM) based on the dominant frequency.
            bpm = 60.0 * hz
            # Store the calculated BPM value in the BPM buffer.
            bpmBuffer[bpmBufferIndex] = bpm
            # Update the index for the BPM buffer, considering the buffer size.
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
              # Append the calculated BPM to the BPM values list.
            bpm_values.append(bpm)

        # Amplify the signal
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha

        # Reconstruct the resulting frame by applying inverse Fourier transformation and filtering.
        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
        # Combine the original face detection frame with the amplified frame.
        outputFrame = detectionFrame + filteredFrame
        # Convert the resulting frame to the appropriate data type for display.
        outputFrame = cv2.convertScaleAbs(outputFrame)
        # Update the buffer index for the processed frames
        bufferIndex = (bufferIndex + 1) % bufferSize
        # Resize the resulting frame for display and update the region of interest on the main frame.
        outputFrame_show = cv2.resize(outputFrame,(videoWidth//2,videoHeight//2))
        frameDraw[0:videoHeight // 2, (realWidth-videoWidth//2):realWidth] = outputFrame_show

        # Calculate the average BPM value from the BPM buffer.
        bpm_value = bpmBuffer.mean()
        imgPlot = plotY.update(float(bpm_value))
        # Display BPM on the frame
        if i > bpmBufferSize:
            cvzone.putTextRect(frameDraw,f'BPM: {bpm_value}',bpmTextLocation,scale=2)
        else:
            cvzone.putTextRect(frameDraw, "Calculating BPM...", loadingTextLocation,scale=2)
        # Display the frame with BPM visualization.
        if len(sys.argv) != 2:
            imgStack = cvzone.stackImages([frameDraw,imgPlot],2,1)
            cv2.imshow("Heart Rate Monitor", imgStack)
            # Check for the 'q' key press to quit the application.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        # If no faces are detected, display the frame without any modifications.
        imgStack = cvzone.stackImages([frameDraw, frameDraw], 2, 1)
        cv2.imshow("Heart Rate Monitor", imgStack)
# Write BPM values to a CSV file
with open('bpm_values.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['BPM'])
    for value in bpm_values:
        writer.writerow([value])
# Clean up
cv2.destroyAllWindows()


