import cv2
import time
import numpy as np
import pyramids
import heartrate
import preprocessing
import eulerian

# Frequency range for Fast-Fourier Transform
freq_min = 1
freq_max = 1.8

# Preprocessing phase
print("Reading + preprocessing video...")
video_path = r"/Users/shrimoysatpathy/Desktop/DriverMVT/With_HR_precise_sync/Driver_1/session_1638006645197/video_1638006628844_1051/testVid.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Failed to open video")
    exit(1)

# Try to get fps from video file metadata
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Detected FPS: {fps}")

video_frames = []
frame_ct = 0

# Capture video with timeout
start_time = time.time()
timeout = 60  # 60 seconds
while True:
    ret, frame = cap.read()
    if not ret:
        break
    video_frames.append(frame)
    frame_ct += 1
    if frame_ct % 180 == 0:
        print("Calculating heart rate for 180 frames...")
        lap_video = pyramids.build_video_pyramid(video_frames[-180:])  # Build pyramid for last 180 frames
        result, fft, frequencies = eulerian.fft_filter(lap_video[1], freq_min, freq_max, fps)  # Assuming appropriate level
        print(f"Size of FFT array: {len(fft)}")
        print(f"Size of frequencies array: {len(frequencies)}")
        max_index = np.argmax(fft)
        print(f"Index from argmax: {max_index}")
        if max_index < len(frequencies):
            print(f"FFT Peak Frequency: {frequencies[max_index]}")  # Debugging: Check the peak frequency
        else:
            print("Error: FFT peak index is out of bounds")
        heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)
        print(f"Heart rate: {heart_rate:.2f} bpm")
    if time.time() - start_time > timeout:
        print("Timeout reached, exiting...")
        break
    if frame_ct == 360:
        break
cap.release()

# Other processing could continue here if needed
