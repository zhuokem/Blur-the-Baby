# pip install opencv-python 
# pip install opencv-python-headless
# !pip install matplotlib
# !pip install retinaface-pytorch
# https://pypi.org/project/retina-face/
# pip install retina-face
# pip install tf-keras
# pip install tensorflow

import cv2
import os
import torch
import time
import pandas as pd
import numpy as np
from retinaface import RetinaFace
import matplotlib.pyplot as plt

# code for extracting frames supported by chatGPT
video_path = directory_path + "Brainy Baby Shapes and Colors.mp4"

# Directory to save the frames
output_dir = directory_path + 'Baby_Original_frames'
# Make the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the timestamp ranges (in seconds) - manually extracted a sample of scenes with babies from watching the video
time_frames = [
    (1 * 60 + 40, 2 * 60 + 5),      # 1:40 to 2:05
    (2 * 60 + 48, 3 * 60 + 10),     # 2:48 to 3:10
    (3 * 60 + 40, 4 * 60),          # 3:40 to 4:00
    (5 * 60 + 35, 5 * 60 + 50),     # 5:35 to 5:50
    (6 * 60 + 13, 6 * 60 + 20),     # 6:13 to 6:20
    (6 * 60 + 35, 6 * 60 + 40),     # 6:35 to 6:40
    (7 * 60 + 20, 7 * 60 + 25),     # 7:20 to 7:25
    (8 * 60 + 5, 8 * 60 + 10),      # 8:05 to 8:10
    (8 * 60 + 35, 8 * 60 + 40),     # 8:35 to 8:40
    (9 * 60 + 10, 9 * 60 + 13),     # 9:10 to 9:13
    (9 * 60 + 20, 9 * 60 + 30),     # 9:20 to 9:30
    (10 * 60, 10 * 60 + 5),         # 10:00 to 10:05
    (10 * 60 + 30, 10 * 60 + 40),   # 10:30 to 10:40
    (10 * 60 + 57, 11 * 60 + 10),   # 10:57 to 11:10
    (11 * 60 + 50, 12 * 60 + 4),    # 11:50 to 12:04
    (12 * 60 + 19, 12 * 60 + 26),   # 12:19 to 12:26
    (13 * 60 + 42, 13 * 60 + 50),   # 13:42 to 13:50
    (14 * 60 + 10, 14 * 60 + 20),   # 14:10 to 14:20
    (14 * 60 + 42, 14 * 60 + 50),   # 14:42 to 14:50
    (14 * 60 + 59, 15 * 60 + 10),   # 14:59 to 15:10
    (15 * 60 + 25, 15 * 60 + 30),   # 15:25 to 15:30
    (16 * 60 + 3, 16 * 60 + 10),    # 16:03 to 16:10 
    (18 * 60 + 5, 18 * 60 + 12),    # 18:05 to 18:12
    (18 * 60 + 36, 18 * 60 + 46),   # 18:36 to 18:46 
    (19 * 60, 19 * 60 + 8),         # 19:00 to 19:08 
    (19 * 60 + 15, 19 * 60 + 23),   # 19:15 to 19:23
    (19 * 60 + 30, 19 * 60 + 40),   # 19:30 to 19:40
    (23 * 60 + 45, 23 * 60 + 50),   # 23:45 to 23:50
    (25 * 60 + 15, 25 * 60 + 20),   # 25:15 to 25:20
    (25 * 60 + 43, 25 * 60 + 50),   # 25:43 to 25:50 
    (26 * 60 + 30, 26 * 60 + 44),   # 26:30 to 26:44 
    (27 * 60 + 20, 27 * 60 + 28),   # 27:20 to 27:28 
    (28 * 60, 28 * 60 + 8),         # 28:00 to 28:08 
    (28 * 60 + 54, 29 * 60 + 3),    # 28:54 to 29:03 
    (29 * 60 + 45, 29 * 60 + 53),   # 29:45 to 29:53
    (30 * 60 + 20, 30 * 60 + 34),   # 30:20 to 30:34
    (30 * 60 + 53, 31 * 60),        # 30:53 to 31:00 
    (31 * 60 + 25, 31 * 60 + 30),   # 31:25 to 31:30
    (32 * 60 + 30, 32 * 60 + 47),   # 32:30 to 32:47
    (34 * 60 + 15, 34 * 60 + 25),   # 34:15 to 34:25 
    (34 * 60 + 48, 35 * 60 + 5),    # 34:48 to 35:05 
    (35 * 60 + 58, 36 * 60 + 5),    # 35:58 to 36:05
    (36 * 60 + 50, 37 * 60 + 2),    # 36:50 to 37:02 
    (37 * 60 + 55, 38 * 60 + 10),   # 37:55 to 38:10
    (41 * 60 + 20, 41 * 60 + 44),   # 41:20 to 41:44
    (42 * 60 + 48, 42 * 60 + 58),   # 42:48 to 42:58
    (43 * 60 + 20, 43 * 60 + 35)    # 43:20 to 43:35
]

# Opens the video file
cap = cv2.VideoCapture(video_path)

# Get the video frame rate (FPS = 29.97)
fps = cap.get(cv2.CAP_PROP_FPS)

# Frame counter
frame_count = 0
saved_frame_count = 0

# Read frames and save the frame if within the specified time frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Check if the current frame falls within any of the ranges identified above
    for start_time, end_time in time_frames:
        start_frame = round(start_time * fps)
        end_frame = round(end_time * fps)

        if start_frame <= frame_count <= end_frame:
            # Save the frame as an image
            frame_filename = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1 # counts the frames selected to be saved
            break

    frame_count += 1 # counts the total frames in the video

cap.release()
print(f"Saved frames for the specified time frames to {output_dir}")
print(f"The video is {fps} frames per second")
print(f"Total frames in the video = {frame_count}")
print(f"Total frames selected = {saved_frame_count}")



def RetinaFace_Detection(image, output_path):
# Detect faces
    detections = RetinaFace.detect_faces(image)

    # Draw bounding boxes and landmarks
    face_count = 0
    for key, face in detections.items():
        facial_area = face["facial_area"]
        x1, y1, x2, y2 = facial_area
        blur_face(image, [x1, y1, x2, y2])  # Blur the face within the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw landmarks. not used.
        # landmarks = face["landmarks"]
        # for _, point in landmarks.items():
        #   cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
        face_count+=1

    cv2.imwrite(output_path, image)
    return face_count


# code gemini support to set the bounding boxes and output the blurred/blocked version

yunet_model = directory_path + 'face_detection_yunet_2023mar.onnx'

# Face detector code comes from YuNet github and chatgpt
image_width = 640
image_height = 480
score_threshold = 0.8 # set default score threshold
nms_threshold = 0.3 # NMS threshold
top_k = 5000 # Maximum faces to retain, realistically would not have this many

face_detector = cv2.FaceDetectorYN_create(yunet_model, "", 
                                          (image_width, image_height), 
                                          score_threshold, 
                                          nms_threshold,
                                            top_k)

# Function to blur the face within the bounding box - ChatGPT sourced
def blur_face(image, box):
    x1, y1, x2, y2 = map(int, box)  # Ensure the coordinates are integers
    # Clip coordinates to stay within the image bounds. ChatGPT sourced to eliminate negative number bounds.
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    # Check if the adjusted bounding box is valid
    if x1 >= x2 or y1 >= y2:
        #print(f"Adjusted invalid bounding box: {x1, y1, x2, y2}")
        return
    face_region = image[y1:y2, x1:x2]
    # Check if the cropped region is valid
    if face_region.size == 0:
        #print(f"Empty face region for adjusted bounding box: {x1, y1, x2, y2}")
        return
    blurred_face = cv2.GaussianBlur(face_region, (51, 51), 30)  # Apply Gaussian blur
    image[y1:y2, x1:x2] = blurred_face  # Replace the face region with the blurred version

yunet_input_size = (320, 320)  # YuNet input size

# we'll set up a dictionary that can keep track of how much each model detects for a given challenge video
yunet_retina_combo_performance = dict()


# Start the timer
start_time = time.time()
output_dir = directory_path + '/Baby_Original_frames'

# counter
frame_count = 0

for img in os.listdir(output_dir):
    if img != ".DS_Store":

        image = cv2.imread(directory_path + '/Baby_Original_frames/' + img)
        image_path = directory_path + '/Baby_Original_frames/' + img
        # Resize to the model's input size
        height, width = image.shape[:2]
        face_detector.setInputSize((width, height))

        # Detect faces
        # go back to using the 0.9 threshold
        _, faces = face_detector.detect(image)

        # Check if faces are detected
        face_count = 0
        if faces is not None:
            
            for face in faces:
                # Extract bounding box coordinates
                x, y, w, h = face[:4]
    
                # code with chatGPT support to figure out how to extract landmarks
                landmarks = face[4:14].reshape(-1, 2)  # Reshape to 5 points (x, y)
                left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    
                confidence = face[-1]
                if confidence < 0.9:
                    face_count = RetinaFace_Detection(image, directory_path + "/ComboV2_babies_detected/"+img)
                else:    
                    # Draw bounding box on the image
                    # source: https://stackoverflow.com/questions/67349410/how-to-draw-filled-rectangle-to-every-frame-of-video-by-using-opencv-python               
                    blur_face(image, [x, y, x + w, y + h])
                    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), -1)

            cv2.imwrite(directory_path + "/ComboV2_babies_detected/"+img, image)   
            num_faces = len(faces)+face_count
            yunet_retina_combo_performance[img] = num_faces
        else:
            face_count = RetinaFace_Detection(image, directory_path + "/ComboV2_babies_detected/"+img)
            num_faces = face_count  
            cv2.imwrite(directory_path + "/ComboV2_babies_detected/"+img, image)    
        
            yunet_retina_combo_performance[img] = num_faces
            
        
        # Increment counter
        frame_count += 1

# End the timer
end_time = time.time()

# Calculate total time taken
total_time = end_time - start_time

# Print total number of blurred frames
print(f"Total number of frames saved: {frame_count}")
print(f"Total time taken: {total_time:.2f} seconds")
