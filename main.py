import cv2
import time
import pyramids
import heartrate
import preprocessing
import eulerian

# Frequency range for Fast-Fourier Transform
freq_min = 1
freq_max = 1.8

# Preprocessing phase
print("Reading + preprocessing video...")
video_path = r"/Users/shrimoysatpathy/Desktop/DriverMVT/With_HR_precise_sync/Driver 1/session_1638006645197/video_1638006628844_1051/video_1638006628844_1051.mp4"
cap = cv2.VideoCapture(video_path)
video_frames = []
frame_ct = 0
fps = 0

# Capture video with timeout
start_time = time.time()
timeout = 60  # 60 seconds
while True:
    ret, frame = cap.read()
    if not ret:
        break
    video_frames.append(frame)
    frame_ct += 1
    if time.time() - start_time > timeout:
        print("Timeout reached, exiting...")
        break
    if frame_ct % 100 == 0:
        print(f"Processed {frame_ct} frames")
    if frame_ct == 500:
        break
cap.release()

# Calculate FPS
if frame_ct > 0:
    elapsed_time = time.time() - start_time
    fps = frame_ct / elapsed_time
    print("Elapsed time:", elapsed_time)
    print("Frame count:", frame_ct)
    print("FPS:", fps)
else:
    print("No frames were captured. Unable to calculate FPS.")
    fps = 0
    
# Build Laplacian video pyramid
print("Building Laplacian video pyramid...")
lap_video = pyramids.build_video_pyramid(video_frames)
amplified_video_pyramid = []

for i, video in enumerate(lap_video):
    if i == 0 or i == len(lap_video)-1:
        continue

    # Eulerian magnification with temporal FFT filtering
    print("Running FFT and Eulerian magnification...")
    result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
    lap_video[i] += result

    # Calculate heart rate every 180 frames
    if frame_ct % 180 == 0:
        print("Calculating heart rate...")
        heart_rate = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max)
        print(f"Heart rate: {heart_rate:.2f} bpm")