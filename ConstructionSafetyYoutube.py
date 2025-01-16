import cv2
import subprocess
import numpy as np
from ultralytics import YOLO
import math

def get_youtube_video_url(youtube_url):
    cmd = ['yt-dlp', '--get-url', '--format', 'best', youtube_url]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return None

def get_video_resolution(video_url):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_url]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        return tuple(map(int, result.stdout.strip().split('x')))
    else:
        return None

# Replace with your YouTube video URL
youtube_url = 'https://www.youtube.com/watch?v=3G6qWa86XwI'
video_url = get_youtube_video_url(youtube_url)

if video_url:
    resolution = get_video_resolution(video_url)
    if resolution:
        width, height = resolution
        ffmpeg_cmd = ['ffmpeg', '-i', video_url, '-f', 'image2pipe', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24', '-']
        pipe = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)

        # Load the YOLO model
        model = YOLO("best.pt")

        # Define class names and colors
        classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
                      'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
                      'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
        myColor = (0, 0, 255)

        bytes_per_frame = width * height * 3
        while True:
            frame = pipe.stdout.read(bytes_per_frame)
            if len(frame) != bytes_per_frame:
                continue  # Incomplete frame, skip

            image = np.frombuffer(frame, dtype=np.uint8).reshape((height, width, 3))
            image = image.copy()  # Ensure the array is writeable

            # Process the frame with YOLO
            results = model(image, stream=True)

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
                        if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0]:
                            cv2.putText(image, f'{classNames[cls]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 0), 2, cv2.LINE_AA)
                            cv2.rectangle(image, (x1, y1), (x2, y2), myColor, 3)

            # Display the image
            cv2.imshow('Video', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pipe.terminate()
        cv2.destroyAllWindows()
    else:
        print("Failed to get video resolution.")
else:
    print("Failed to get video URL.")