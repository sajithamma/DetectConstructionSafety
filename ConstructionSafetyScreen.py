from ultralytics import YOLO
import cv2
import numpy as np
from mss import mss
import math
from helper import create_video_writer

# Set up screen capture
monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Adjust for your screen resolution
sct = mss()

# Set up video writer
frame_width = monitor["width"]
frame_height = monitor["height"]
writer = cv2.VideoWriter("ConstructionSiteSafetyOutput.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))

model = YOLO("best.pt")

classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
              'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
              'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
myColor = (0, 0, 255)

while True:
    # Capture screen
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR

    # Process frame with YOLO
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
                if currentClass =='NO-Hardhat' or currentClass =='NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0,255)
                elif currentClass =='Hardhat' or currentClass =='Safety Vest' or currentClass == "Mask":
                    myColor =(0,255,0)
                else:
                    myColor = (255, 0, 0)

                image = cv2.putText(frame, f'{classNames[cls]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 3)

    # Display the frame
    cv2.imshow("Screen Capture", frame)
    writer.write(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources
writer.release()
cv2.destroyAllWindows()