import torch
import sqlite3
from facenet_pytorch import InceptionResnetV1, MTCNN
import os

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
