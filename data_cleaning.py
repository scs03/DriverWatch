import cv2
import numpy as np
from scipy.interpolate import interp1d
from preprocessing import extract_faces
import pandas as pd
import os

# Specify the directory containing the videos and CSV files
data_dir = '/Users/vasudevnair113/Downloads/DriverWatch/vishnucode/vishnucode2/GDdata'

# Get a list of all video and CSV files in the directory
video_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mp4')])
csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

ihr = []
for video_file, csv_file in zip(video_files, csv_files):
    # Load the array of heart rates from the CSV file
    df = pd.read_csv(os.path.join(data_dir, csv_file), encoding='utf-8-sig')
    heart_rates = df.iloc[:, 0].values

    # Load the video to determine frame rate
    video_path = os.path.join(data_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in the video
    cap.release()

    # Total video duration in seconds
    duration_seconds = total_frames / fps

    # Define a new set of time points that matches the video duration exactly
    # Assuming you process every 6 frames
    frame_times = np.arange(0, total_frames, 6) / fps  # One value for each batch of 6 frames
    m = len(heart_rates)  # Number of heart rate entries
    time_points = np.linspace(0, duration_seconds, num=m)  # Original time points for heart rate values

    # Create an interpolation function with extrapolation to cover all frame times
    heart_rate_interpolation = interp1d(time_points, heart_rates, kind='linear', fill_value="extrapolate")

    # Interpolate heart rates at these new frame times
    interpolated_heart_rates = heart_rate_interpolation(frame_times)
    ihr.append(interpolated_heart_rates)
    # Print each batch's corresponding heart rate
    for i, rate in enumerate(interpolated_heart_rates):
        print(f"Heart rate for frames {6*i} to {6*(i+1)-1} in video {video_file}: {rate:.2f}")
