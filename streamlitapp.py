from ultralytics import YOLO
import cv2
import math
import tempfile
import streamlit as st
import numpy as np
from PIL import Image
import os
import subprocess
import face_recognition  # For face recognition

# Load the YOLO model
model = YOLO("best.pt")

# Define class names and colors
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
              'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
myColor = (0, 0, 255)

# Streamlit app
st.title('Zaper Safety - Live Video Safety Monitoring')

# Sidebar for input options
st.sidebar.title("Input Options")
input_option = st.sidebar.radio("Choose input source:", ("Preset Videos", "Upload Video", "Camera Capture", "YouTube URL"))

# Face recognition setup
st.sidebar.title("Face Recognition")
uploaded_faces = st.sidebar.file_uploader("Upload face photos (one at a time)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
face_names = st.sidebar.text_input("Enter the name of the person (comma-separated for multiple faces):")

# Store face encodings and names
known_face_encodings = []
known_face_names = []

if uploaded_faces and face_names:
    face_names_list = [name.strip() for name in face_names.split(",")]
    if len(uploaded_faces) != len(face_names_list):
        st.sidebar.error("Number of uploaded faces must match the number of names provided.")
    else:
        for i, uploaded_face in enumerate(uploaded_faces):
            face_image = face_recognition.load_image_file(uploaded_face)
            face_encoding = face_recognition.face_encodings(face_image)
            if len(face_encoding) > 0:
                known_face_encodings.append(face_encoding[0])
                known_face_names.append(face_names_list[i])
            else:
                st.sidebar.warning(f"No face detected in {uploaded_face.name}. Skipping.")

# Function to process frames with face recognition
def process_frame_with_faces(frame):
    # Ensure the frame is writeable
    frame = frame.copy()

    # Perform YOLO object detection
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

    # Perform face recognition
    if len(known_face_encodings) > 0:
        # Find all face locations and encodings in the frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare the face with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)

            # Draw a label with the name below the face
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    return frame

# Function to list available camera devices
def list_camera_devices():
    devices = []
    for index in range(10):  # Check up to 10 devices
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            devices.append(f"Camera {index}")
            cap.release()
        else:
            cap.release()
    return devices

# Function to list preset videos in the 'videos' folder
def list_preset_videos():
    video_folder = "videos"  # Folder containing preset videos
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)
        st.warning(f"No videos found in the '{video_folder}' folder. Please add some videos.")
        return []
    videos = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
    return videos

# Function to get YouTube video URL
def get_youtube_video_url(youtube_url):
    cmd = ['yt-dlp', '--get-url', '--format', 'best', youtube_url]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return None

# Function to get video resolution
def get_video_resolution(video_url):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_url]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return tuple(map(int, result.stdout.strip().split('x')))
    else:
        return None

# Option 1: Upload Video
if input_option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
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

            # Process the frame with face recognition
            processed_frame = process_frame_with_faces(frame)

            # Write the processed frame
            out.write(processed_frame)

            # Display the processed frame in Streamlit
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_placeholder.image(frame_pil, use_container_width=True)

        # Release resources
        cap.release()
        out.release()

        # Display the processed video
        st.write('Processed Video:')
        video_bytes = open(output_temp_file.name, 'rb').read()
        st.video(video_bytes)

# Option 2: Camera Capture
elif input_option == "Camera Capture":
    st.sidebar.write("Using camera for real-time processing...")

    # List available cameras
    camera_devices = list_camera_devices()
    if not camera_devices:
        st.error("No cameras found.")
    else:
        # Dropdown to select camera device
        selected_camera = st.sidebar.selectbox("Select Camera Device:", camera_devices)
        camera_index = int(selected_camera.split(" ")[1])  # Extract camera index from the selected option

        # Initialize camera
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            st.error(f"Unable to access {selected_camera}.")
        else:
            st.write(f"Real-time processing using {selected_camera}...")
            frame_placeholder = st.empty()

            # Add a button to stop the camera feed
            stop_camera = st.sidebar.button("Stop Camera")

            while True:
                if stop_camera:
                    st.write("Camera feed stopped.")
                    break

                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame.")
                    break

                # Process the frame with face recognition
                processed_frame = process_frame_with_faces(frame)

                # Display the processed frame in Streamlit
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_placeholder.image(frame_pil, use_container_width=True)

                # Break the loop if 'q' is pressed (for local testing)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release resources
            cap.release()

# Option 3: Preset Videos
elif input_option == "Preset Videos":
    st.sidebar.write("Select a preset video for processing...")

    # List preset videos
    preset_videos = list_preset_videos()
    if not preset_videos:
        st.error("No preset videos found in the 'videos' folder.")
    else:
        # Dropdown to select a preset video
        selected_video = st.sidebar.selectbox("Select Preset Video:", preset_videos)
        video_path = os.path.join("videos", selected_video)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            st.error(f"Unable to open {selected_video}.")
        else:
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Desired FPS for processing and output
            desired_fps = 10  # Adjust this value as needed
            frame_skip_interval = int(fps / desired_fps)

            # Create a temporary output file
            output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

            # Create VideoWriter for output with desired FPS
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_temp_file.name, fourcc, desired_fps, (width, height))

            # Read frames and process
            st.write(f'Processing {selected_video}...')

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

                # Process the frame with face recognition
                processed_frame = process_frame_with_faces(frame)

                # Write the processed frame
                out.write(processed_frame)

                # Display the processed frame in Streamlit
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_placeholder.image(frame_pil, use_container_width=True)

            # Release resources
            cap.release()
            out.release()

            # Display the processed video
            st.write('Processed Video:')
            video_bytes = open(output_temp_file.name, 'rb').read()
            st.video(video_bytes)

# Option 4: YouTube URL
elif input_option == "YouTube URL":
    st.sidebar.write("Enter a YouTube URL for processing...")

    # Input for YouTube URL
    youtube_url = st.sidebar.text_input("Enter YouTube URL:")
    if youtube_url:
        # Get the video URL using yt-dlp
        video_url = get_youtube_video_url(youtube_url)
        if video_url:
            # Get video resolution
            resolution = get_video_resolution(video_url)
            if resolution:
                width, height = resolution

                # Create a temporary output file
                output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

                # Create VideoWriter for output with desired FPS
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_temp_file.name, fourcc, 10, (width, height))  # Default FPS: 10

                # FFmpeg command to read the video stream
                ffmpeg_cmd = ['ffmpeg', '-i', video_url, '-f', 'image2pipe', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-']
                pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)

                # Read frames and process
                st.write('Processing YouTube video...')

                frame_placeholder = st.empty()
                bytes_per_frame = width * height * 3

                while True:
                    frame = pipe.stdout.read(bytes_per_frame)
                    if len(frame) != bytes_per_frame:
                        break  # End of video

                    # Convert frame to numpy array
                    image = np.frombuffer(frame, dtype=np.uint8).reshape((height, width, 3))

                    # Ensure the frame is writeable
                    image = image.copy()

                    # Process the frame with face recognition
                    processed_frame = process_frame_with_faces(image)

                    # Write the processed frame
                    out.write(processed_frame)

                    # Display the processed frame in Streamlit
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_placeholder.image(frame_pil, use_container_width=True)

                # Release resources
                pipe.terminate()
                out.release()

                # Display the processed video
                st.write('Processed Video:')
                video_bytes = open(output_temp_file.name, 'rb').read()
                st.video(video_bytes)
            else:
                st.error("Failed to get video resolution.")
        else:
            st.error("Failed to get video URL.")