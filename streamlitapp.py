from ultralytics import YOLO
import cv2
import math
import tempfile
import streamlit as st
import numpy as np
from PIL import Image

# Load the YOLO model
model = YOLO("best.pt")

# Define class names and colors
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
              'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
myColor = (0, 0, 255)

# Streamlit app
st.title('Construction Site Safety Video Processing')

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Create a temporary file to save the uploaded content
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Create a temporary output file
    output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

    # Open the video file
    cap = cv2.VideoCapture(tfile.name)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Desired FPS for processing and output
    desired_fps = 10  # Adjust this value as needed
    frame_skip_interval = int(fps / desired_fps)

    # Create VideoWriter for output with desired FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_temp_file.name, fourcc, desired_fps, (width, height))

    # Read frames and process
    st.write('Processing video...')

    frame_placeholder = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to reduce FPS
        frame_count += 1
        if frame_count % frame_skip_interval != 0:
            continue

        # Process the frame
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                print(currentClass)
                if conf > 0.5:
                    if currentClass in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                        myColor = (0, 0, 255)
                    elif currentClass in ['Hardhat', 'Safety Vest', 'Mask']:
                        myColor = (0, 255, 0)
                    else:
                        myColor = (255, 0, 0)

                    # Ensure coordinates are within image bounds
                    if 0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0]:
                        cv2.putText(frame, f'{classNames[cls]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 3)

        # Write the processed frame
        out.write(frame)

        # Display the processed frame in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_placeholder.image(frame_pil, use_container_width=True)

    # Release resources
    cap.release()
    out.release()

    # Display the processed video
    st.write('Processed Video:')
    video_bytes = open(output_temp_file.name, 'rb').read()
    st.video(video_bytes)