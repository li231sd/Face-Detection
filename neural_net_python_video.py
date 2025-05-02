import tkinter as tk
from tkinter import ttk
import threading
import time
import math
import random
import cv2
from PIL import Image, ImageTk
import requests
import folium
import io
from geopy.geocoders import Nominatim
import numpy as np
import speech_recognition as sr
import pyaudio
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import tkinter.messagebox

class FaceRecognitionModel(nn.Module):
    def __init__(self):
        super(FaceRecognitionModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Two classes: known, unknown face

    def forward(self, x):
        return self.model(x)

class Robot:
    def __init__(self):
        self.lat, self.lon = 37.7749, -122.4194  # Default start (San Francisco)
        self.heading = 0
        self.track = [(self.lat, self.lon)]
        self.motor_left_speed = 0
        self.motor_right_speed = 0
        self.steering_pid = PID(2.0, 0.0, 1.0)
        self.destination = None
        self.running = False

    def simulate_movement(self):
        avg_speed = (self.motor_left_speed + self.motor_right_speed) / 2
        self.lat += avg_speed * 0.00001 * math.cos(math.radians(self.heading))
        self.lon += avg_speed * 0.00001 * math.sin(math.radians(self.heading))
        self.track.append((self.lat, self.lon))

    def update_heading(self, steering_correction):
        self.heading += steering_correction
        self.heading %= 360

    def set_destination(self, place_name):
        geolocator = Nominatim(user_agent="robot_navigator")
        location = geolocator.geocode(place_name)
        if location:
            self.destination = (location.latitude, location.longitude)
        else:
            self.destination = None

    def compute_drive(self):
        if not self.destination:
            return
        target_lat, target_lon = self.destination
        error_lat = target_lat - self.lat
        error_lon = target_lon - self.lon
        desired_heading = math.degrees(math.atan2(error_lon, error_lat))
        heading_error = (desired_heading - self.heading + 540) % 360 - 180

        steering = self.steering_pid.compute(heading_error, dt=0.1)

        base_speed = 1.0
        left_speed = base_speed - steering * 0.05
        right_speed = base_speed + steering * 0.05

        self.motor_left_speed = max(min(left_speed, 2.0), -2.0)
        self.motor_right_speed = max(min(right_speed, 2.0), -2.0)

    def detect_obstacle(self):
        return random.random() < 0.05

    def avoid_obstacle(self):
        self.heading += 90
        self.heading %= 360

# PID Controller
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

# Main Dashboard GUI with Tkinter
class RobotApp:
    def __init__(self, root, robot, face_recognition_model):
        self.root = root
        self.robot = robot
        self.face_recognition_model = face_recognition_model
        self.unknown_face_counter = 0  # Track how many times an unknown face is detected
        self.unknown_face_limit = 3     # Limit after which the program will exit

        self.root.title("Robot Dashboard")

        # Mode selection buttons
        self.gps_button = tk.Button(root, text="GPS Navigation", command=self.start_navigation)
        self.gps_button.pack(pady=10)

        self.manual_button = tk.Button(root, text="Manual Mode", command=self.manual_mode)
        self.manual_button.pack(pady=10)

        # Exit button
        self.exit_button = tk.Button(root, text="Exit", command=self.on_close)
        self.exit_button.pack(pady=10)

        # Video frame
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        self.cap = cv2.VideoCapture(0)
        self.update_video()

        self.update_map()

        # Start voice command in background
        threading.Thread(target=self.listen_for_voice_command, daemon=True).start()

    def on_close(self):
        # Stop the webcam and close the Tkinter window
        self.cap.release()
        self.root.quit()
        self.root.destroy()

    def start_navigation(self):
        place = "Golden Gate Bridge"  
        self.robot.set_destination(place)
        if self.robot.destination:
            self.robot.running = True
            threading.Thread(target=self.drive_loop, daemon=True).start()

    def manual_mode(self):
        print("Running Manual Control Mode...")

    def drive_loop(self):
        while self.robot.running:
            if self.robot.detect_obstacle():
                print("Obstacle detected! Avoiding...")
                self.robot.avoid_obstacle()
            else:
                self.robot.compute_drive()
            self.robot.simulate_movement()
            time.sleep(0.1)

    def update_map(self):
        self.root.after(500, self.update_map)

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.face_recognition(frame_rgb)  # Perform face recognition

            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(30, self.update_video)

    def face_recognition(self, frame):
        # Convert frame to PyTorch tensor
        transform = transforms.Compose([  
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        ])
        
        # Convert image to tensor
        input_tensor = transform(frame)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.face_recognition_model.to('cuda')

        with torch.no_grad():
            output = self.face_recognition_model(input_batch)

        _, predicted = torch.max(output, 1)
        if predicted == 0:
            print("Known face recognized!")
        else:
            print("Unknown face detected.")
            self.unknown_face_counter += 1
            if self.unknown_face_counter >= self.unknown_face_limit:
                self.show_unknown_face_alert()
                self.on_close()

    def show_unknown_face_alert(self):
        tk.messagebox.showwarning("Unknown Face Detected", "An unknown face has been detected. The program will exit.")

    def listen_for_voice_command(self):
        # Start listening for voice command in the background
        command = voice_command()
        if command == "gps":
            self.start_navigation()
        elif command == "manual":
            self.manual_mode()
        else:
            print("No valid voice command detected.")

# Voice Mode Selection
def voice_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for mode selection...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"Voice Command: {command}")

        if "gps" in command:
            return "gps"
        elif "manual" in command:
            return "manual"
        else:
            print("Invalid command. Please say 'GPS' or 'Manual'.")
            return None
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Error with speech recognition: {e}")
        return None

if __name__ == "__main__":
    root = tk.Tk()
    face_recognition_model = FaceRecognitionModel()
    robot = Robot()
    app = RobotApp(root, robot, face_recognition_model)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
