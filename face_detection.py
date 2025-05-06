import cv2
import mediapipe as mp
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
import gridfs
import io
import certifi

# MongoDB setup
client = MongoClient("mongodb+srv://scott:1eeypCNUmvjA6i72@drone.ptmiqhu.mongodb.net/", tlsCAFile=certifi.where())
db = client["face_recognition_db"]
collection = db["faces"]
fs = gridfs.GridFS(db)

# Paths
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
cap = cv2.VideoCapture(0)

# Globals
face_visible_since = None
image_saved = False
THRESHOLD_SECONDS = 3
MODEL_PATH = "face_classifier.h5"
IMG_SIZE = 128
model = None
label_encoder = LabelEncoder()

# Load dataset
def load_data_from_mongo():
    X, y = [], []
    try:
        # Fetch all documents with faces and encodings
        for doc in db.faces.find():
            name = doc['name']
            image_data = fs.get(doc['image']).read()
            img_array = np.frombuffer(image_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Resize image to a fixed size (IMG_SIZE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Normalize image to the range [0, 1]
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(name)
        
        print(f"✅ Loaded {len(X)} faces and names from MongoDB")
    except Exception as e:
        print(f"❌ Error loading data from MongoDB: {e}")
    
    return np.array(X), np.array(y)

# Create a simple CNN model
def create_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the model
def train_model():
    global model, label_encoder
    X, y = load_data_from_mongo()
    if len(set(y)) < 2:
        print("⚠️ Need at least 2 people to train the model.")
        return
    label_encoder.fit(y)
    y_encoded = label_encoder.transform(y)
    X_train, _, y_train, _ = train_test_split(X, y_encoded, test_size=0.2)
    model = create_model(len(set(y)))
    model.fit(X_train, y_train, epochs=10, verbose=0)
    model.save(MODEL_PATH)
    print("✅ Model trained and saved.")

# Predict face
def predict_person(img_crop):
    if model is None:
        return "unknown"
    img = cv2.resize(img_crop, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img / 255.0, axis=0)
    pred = model.predict(img, verbose=0)[0]
    class_index = np.argmax(pred)
    confidence = pred[class_index]
    if confidence < 0.8:
        return "unknown"
    return label_encoder.inverse_transform([class_index])[0]

# Save to MongoDB
def save_face_to_mongo(face_crop, name):
    _, buffer = cv2.imencode('.jpg', face_crop)
    img_binary = buffer.tobytes()
    user_data = {
        "name": name,
        "image": fs.put(img_binary, filename=f"{name}.jpg")
    }
    collection.insert_one(user_data)
    print(f"✅ {name}'s face saved to MongoDB.")

# Auto-load or create model
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded from file.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
else:
    print("⚠️ Model file not found. Training new model...")
    try:
        train_model()
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ New model trained and loaded.")
    except Exception as e:
        print(f"❌ Could not train model: {e}")

# Main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    state = "idle"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            x1, y1 = int(min(xs) * w), int(min(ys) * h)
            x2, y2 = int(max(xs) * w), int(max(ys) * h)
            x1, y1 = max(0, x1 - 20), max(0, y1 - 20)
            x2, y2 = min(w, x2 + 20), min(h, y2 + 20)

            if face_visible_since is None:
                face_visible_since = time.time()
            elif time.time() - face_visible_since >= THRESHOLD_SECONDS and not image_saved:
                state = "capturing"
                face_crop = frame[y1:y2, x1:x2]
                cv2.imshow("Captured Face", face_crop)
                name = input("Enter name for captured face: ").strip()
                save_path = os.path.join(DATASET_DIR, name)
                os.makedirs(save_path, exist_ok=True)
                count = len(os.listdir(save_path)) + 1
                cv2.imwrite(os.path.join(save_path, f"{count}.jpg"), face_crop)
                save_face_to_mongo(face_crop, name)
                image_saved = True
                train_model()
                if os.path.exists(MODEL_PATH):
                    model = tf.keras.models.load_model(MODEL_PATH)
            else:
                state = "countdown"

            if state == "capturing":
                color = (0, 255, 0)  # Green
            else:
                face_crop = frame[y1:y2, x1:x2]
                person_name = predict_person(face_crop)
                if person_name != "unknown":
                    color = (255, 0, 0)  # Blue
                    print(f"✅ Recognized: {person_name}")
                else:
                    color = (0, 0, 255)  # Red

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    else:
        face_visible_since = None
        image_saved = False

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
