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
realWidth = 640   
realHeight = 480  
videoWidth = 160
videoHeight = 120
videoChannels = 3
videoFrameRate = 15

# Initialize face detector and BPM values list
detector = FaceDetector()
# List to store BPM values
bpm_values = []  

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# initialize live plot for BPM visulaization
plotY = LivePlot(realWidth,realHeight,[60,120],invert=True)

# Helper Methods
#construct Gaussian pyramid for the given frame
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid
# Reconstruct frame from the Gaussian pyramid
def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame


# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (30, 40)
bpmTextLocation = (videoWidth//2, 40)
fpsTextLoaction = (500,600)

# initialize graph items
fontScale = 1
fontColor = (255,255,255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 10   #15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

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
        x1, y1, w1, h1 = bboxs[0]['bbox']
        cv2.rectangle(frameDraw, bboxs[0]['bbox'], (255, 0, 255), 2)
        detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
        detectionFrame = cv2.resize(detectionFrame,(videoWidth,videoHeight))

        # Construct Gaussian Pyramid for the detected face
        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
        fourierTransform = np.fft.fft(videoGauss, axis=0)

        # Apply bandpass filter to the Fourier transform
        fourierTransform[mask == False] = 0

        # Extract heart rate information
        if bufferIndex % bpmCalculationFrequency == 0:
            i = i + 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
            bpm_values.append(bpm)

        # Amplify the signal
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha

        # Reconstruct Resulting Frame
        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        bufferIndex = (bufferIndex + 1) % bufferSize
        outputFrame_show = cv2.resize(outputFrame,(videoWidth//2,videoHeight//2))
        frameDraw[0:videoHeight // 2, (realWidth-videoWidth//2):realWidth] = outputFrame_show

        bpm_value = bpmBuffer.mean()
        imgPlot = plotY.update(float(bpm_value))
        # Display BPM on the frame
        if i > bpmBufferSize:
            cvzone.putTextRect(frameDraw,f'BPM: {bpm_value}',bpmTextLocation,scale=2)
        else:
            cvzone.putTextRect(frameDraw, "Calculating BPM...", loadingTextLocation,scale=2)
        # Display frame with BPM visualization
        if len(sys.argv) != 2:
            imgStack = cvzone.stackImages([frameDraw,imgPlot],2,1)
            cv2.imshow("Heart Rate Monitor", imgStack)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
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


