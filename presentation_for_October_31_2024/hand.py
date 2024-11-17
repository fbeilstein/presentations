import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import simpledialog


def get_ip():
    # Create a tkinter root window (it will be hidden)
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    # Show an input dialog
    ip_address = simpledialog.askstring("HAND", "Enter the IP address of the device to connect to:")

    # Destroy the root window after input is done
    root.destroy()
    return ip_address


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


if True:
    cap = cv2.VideoCapture(0)
    koef = 1.0
else:
    ip = get_ip()
    cap = cv2.VideoCapture(f"rtsp://{ip}:554/axis-media/media.amp")
    koef = 0.25


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    image = cv2.resize(image, (0, 0), fx=koef, fy=koef)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

    for _ in range(20): 
      cap.grab()

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()