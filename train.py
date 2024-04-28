import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from preprocessing import extract_faces
import numpy as np
from data_cleaning import ihr
import os

# Specify the directory containing the videos and CSV files
data_dir = "/root/vishnucode2/GDdata"

# Get a list of all video and CSV files in the directory
video_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mp4')])

def create_model(input_shape):
    model = Sequential([
        Conv3D(32, (3, 3, 3), activation='relu', input_shape=input_shape),
        MaxPooling3D((2, 2, 2)),
        Conv3D(64, (2, 2, 2), activation='relu'),
        MaxPooling3D((1, 2, 2)),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(faces, heart_rates, sequence_length=6):
    X, y = [], []
    for i in range(0, len(faces) - sequence_length + 1, sequence_length):
        X.append(faces[i:i+sequence_length])
        if len(heart_rates[i:i+sequence_length]) > 0:  # Check if there are enough heart rate values
            y.append(np.mean(heart_rates[i:i+sequence_length]))  # averaging heart rate over the sequence
        else:
            print("Insufficient heart rate data.")
    return np.array(X), np.array(y)

X_train_all, y_train_all = [], []

counter = 0
for video_file in video_files:
    video_path = os.path.join(data_dir, video_file)
    faces = extract_faces(video_path)
    heart_rates = ihr[counter]

    # Prepare data for each video and CSV file
    X_train, y_train = prepare_data(faces, heart_rates)
    X_train_all.extend(X_train)
    y_train_all.extend(y_train)
    counter += 1


# Convert lists to numpy arrays
X_train_all = np.array(X_train_all)
y_train_all = np.array(y_train_all)

input_shape = (6, 224, 224, 3)  # 4 is the sequence length, 224x224 is the image resolution, 3 stands for RGB channels
model = create_model(input_shape)
model.fit(X_train_all, y_train_all, epochs=3, batch_size=32, validation_split=0.2)