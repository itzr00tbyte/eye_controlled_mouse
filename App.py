import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import tkinter as tk
from threading import Thread

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Initialize the camera
cam = cv2.VideoCapture(0)

# Screen size for mouse movement
screen_w, screen_h = pyautogui.size()

# Global variables for cursor smoothing
smoothness_factor = 5
values_buffer_x, values_buffer_y = [], []

# Function to calculate moving average for smooth movement
def moving_average(new_value, values_buffer):
    values_buffer.append(new_value)
    if len(values_buffer) > smoothness_factor:
        values_buffer.pop(0)
    return sum(values_buffer) / len(values_buffer)

# Main function for eye tracking and control
def eye_control():
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks

        if landmark_points:
            landmarks = landmark_points[0].landmark

            for id, landmark in enumerate(landmarks[474:478]):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                if id == 1:
                    smooth_x = moving_average(x, values_buffer_x)
                    smooth_y = moving_average(y, values_buffer_y)
                    screen_x = screen_w * landmark.x
                    screen_y = screen_h * landmark.y
                    pyautogui.moveTo(screen_x, screen_y)

            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 255))

            if (left[0].y - left[1].y) < 0.004:
                pyautogui.click()
                cv2.putText(frame, "Clicked", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Eye Controlled Mouse', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# GUI for settings (Placeholder for future development)
def settings_gui():
    root = tk.Tk()
    root.title("Settings")

    # Add settings controls here

    root.mainloop()

# Running the eye control in a separate thread
eye_thread = Thread(target=eye_control)
eye_thread.start()

# Running the settings GUI in the main thread
settings_gui()

# Ensure everything is closed properly when done
eye_thread.join()
