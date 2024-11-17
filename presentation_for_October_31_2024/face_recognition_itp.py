import face_recognition
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import simpledialog


def get_ip():
    # Create a tkinter root window (it will be hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    # Show an input dialog
    ip_address = simpledialog.askstring("FACE", "Enter the IP address of the device to connect to:")

    # Destroy the root window after input is done
    root.destroy()
    return ip_address


font_path = "C:/Windows/Fonts/arial.ttf"
font = ImageFont.truetype(font_path, 32)

# Load images of your coworkers
known_images = []
known_names = []

with open('db_small.txt', 'r', encoding="utf-8") as db:
    for line in db.readlines():
        if not line:
            continue
        url, name = line.split('\t')
        known_images.append(url)
        known_names.append(name)
        #print(name, url)

# Encode known images
known_encodings = []
for img_path,name in zip(known_images, known_names):
    img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_img.astype('uint8'))
    if encodings:
        print(name)
        print(encodings[0])
        known_encodings.append(encodings[0])



if True:
    video_capture = cv2.VideoCapture(0)
    koef = 1.0
else:
    ip = get_ip()
    video_capture = cv2.VideoCapture(f"rtsp://{ip}:554/axis-media/media.amp")
    koef = 0.25


while True:
    ret, frame = video_capture.read()
    if not ret:
        continue
    frame = cv2.resize(frame, (0, 0), fx=koef, fy=koef)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype('uint8')

    # Find faces and get encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Compare each detected face with known faces
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        # If a match was found, get the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Draw label around face
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)
        draw.text((left, top - 10), name, font=font, fill=(255, 255, 255))
        frame = np.array(pil_img)

        #cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Video", frame)
    for _ in range(20):
        video_capture.grab()


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
