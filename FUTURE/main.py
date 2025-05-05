import tkinter as tk
from tkinter import messagebox
import threading
import time
import math
import random
import cv2
from PIL import Image, ImageTk
from geopy.geocoders import Nominatim
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import speech_recognition as sr
import sqlite3
import os

# --- PID Controller ---
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

# --- Robot Class (directly in main.py) ---
class Robot:
    def __init__(self):
        self.lat, self.lon = 37.7749, -122.4194
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
            print(f"Destination set: {self.destination}")
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

# --- Face Recognition Class ---
class FaceRecognition:
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.known_embeddings = {}
        self.db_path = "faces.db"
        self.initialize_database()

    def initialize_database(self):
        if not os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute('''CREATE TABLE faces (name TEXT, embedding BLOB)''')
            conn.commit()
            conn.close()

    def register_face(self, name, frame):
        face = self.mtcnn(frame)
        if face is not None:
            with torch.no_grad():
                embedding = self.embedder(face.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu'))
            self.known_embeddings[name] = embedding
            self.save_face_to_db(name, embedding)
            print(f"Face {name} registered.")
        else:
            print("No face detected.")

    def save_face_to_db(self, name, embedding):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO faces (name, embedding) VALUES (?, ?)", (name, embedding.numpy().tobytes()))
        conn.commit()
        conn.close()

    def recognize_face(self, frame):
        face = self.mtcnn(frame)
        if face is not None:
            with torch.no_grad():
                embedding = self.embedder(face.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu'))
            name, distance = self.match_face(embedding)
            return name, distance
        return None, None

    def match_face(self, embedding):
        min_distance = float('inf')
        matched_name = None
        for name, known_embedding in self.known_embeddings.items():
            distance = torch.nn.functional.pairwise_distance(embedding, known_embedding).item()
            if distance < min_distance:
                min_distance = distance
                matched_name = name
        return matched_name, min_distance

# --- Voice Control Class ---
class VoiceControl:
    def __init__(self, robot):
        self.robot = robot
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

    def listen_for_commands(self):
        while True:
            with self.microphone as source:
                print("Listening for voice commands...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source)
            try:
                command = self.recognizer.recognize_google(audio).lower()
                print(f"Recognized command: {command}")
                self.execute_command(command)
            except sr.UnknownValueError:
                print("Sorry, I didn't understand that.")
            except sr.RequestError:
                print("Sorry, there was an issue with the speech recognition service.")

    def execute_command(self, command):
        if "start" in command:
            print("Starting robot...")
            self.robot.running = True
            threading.Thread(target=self.robot.drive_loop, daemon=True).start()
        elif "stop" in command:
            print("Stopping robot...")
            self.robot.running = False
        elif "set destination" in command:
            print("Setting new destination...")
            # Here you could add parsing of the destination from the voice command
            # and call `self.robot.set_destination(destination)`
        else:
            print("Command not recognized.")

# --- Main Dashboard GUI ---
class RobotApp:
    def __init__(self, root, robot, face_recognition, voice_control):
        self.root = root
        self.robot = robot
        self.face_recognition = face_recognition
        self.voice_control = voice_control
        self.root.title("Robot Dashboard")

        self.gps_button = tk.Button(root, text="GPS Navigation", command=self.start_navigation)
        self.gps_button.pack(pady=10)

        self.manual_button = tk.Button(root, text="Manual Mode", command=self.manual_mode)
        self.manual_button.pack(pady=10)

        self.register_button = tk.Button(root, text="Register Face", command=self.register_face)
        self.register_button.pack(pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=self.exit_program)
        self.exit_button.pack(pady=10)

        self.destination_label = tk.Label(root, text="Enter destination:")
        self.destination_label.pack(pady=10)

        self.destination_entry = tk.Entry(root)
        self.destination_entry.pack(pady=10)

        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        self.cap = cv2.VideoCapture(0)
        self.update_video()

    def start_navigation(self):
        destination = self.destination_entry.get()
        if destination:
            self.robot.set_destination(destination)
            if self.robot.destination:
                self.robot.running = True
                threading.Thread(target=self.drive_loop, daemon=True).start()

    def manual_mode(self):
        print("Running Manual Control Mode...")

    def drive_loop(self):
        def run_drive():
            while self.robot.running:
                if self.robot.detect_obstacle():
                    print("Obstacle detected! Avoiding...")
                    self.robot.avoid_obstacle()
                else:
                    self.robot.compute_drive()
                self.robot.simulate_movement()
                time.sleep(0.1)

        threading.Thread(target=run_drive, daemon=True).start()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_recognition.recognize_face(frame_rgb)
            print(result)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(30, self.update_video)

    def register_face(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame.")
            return
        result = self.face_recognition.register_face(frame)
        messagebox.showinfo("Face Registration", result)

    def exit_program(self):
        self.cap.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    robot = Robot()  # Robot class is now directly inside this file
    face_recognition = FaceRecognition()  # Face recognition setup
    voice_control = VoiceControl(robot)  # Voice control setup
    app = RobotApp(root, robot, face_recognition, voice_control)
    
    # Start listening for voice commands in a separate thread
    threading.Thread(target=voice_control.listen_for_commands, daemon=True).start()

    root.protocol("WM_DELETE_WINDOW", app.exit_program)
    root.mainloop()
