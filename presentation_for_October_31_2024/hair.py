
import cv2
import numpy as np
from fire_hair_utils import HairSegmentation, get_fire_gif
import tkinter as tk
from tkinter import simpledialog


def get_ip():
    # Create a tkinter root window (it will be hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    # Show an input dialog
    ip_address = simpledialog.askstring("FIRE", "Enter the IP address of the device to connect to:")

    # Destroy the root window after input is done
    root.destroy()
    return ip_address


if True:
    cap = cv2.VideoCapture(0)
    webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
else:
    ip = get_ip()
    cap = cv2.VideoCapture(f"rtsp://{ip}:554/axis-media/media.amp")
    webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//4
    webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//4

cv2.namedWindow("Fire Hair", cv2.WINDOW_NORMAL)

# Inialize hair segmentation model
hair_segmentation = HairSegmentation(webcam_width, webcam_height)

while cap.isOpened():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Read frame
    ret, frame = cap.read()
    if not ret:
        continue

    img_height, img_width, _ = frame.shape
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Flip the image horizontally
    frame = cv2.flip(frame, 1)

    # Segment hair
    hair_mask = hair_segmentation(frame)

    # Draw fire 
    combined_image = hair_segmentation.draw_fire_hair(frame, hair_mask)

    cv2.imshow("Fire Hair", combined_image)
    for _ in range(20):
        cap.grab()
