import os
import cv2
import face_recognition
import numpy as np
import tensorflow as tf
from datetime import datetime

# Directory to store known face encodings and training images
known_faces_dir = 'known_faces'
train_dir = 'training_images'

if not os.path.exists(known_faces_dir):
    os.makedirs(known_faces_dir)

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

# Check if the saved face database exists, and load it if possible
def load_face_database():
    try:
        known_face_encodings = np.load(os.path.join(known_faces_dir, 'known_face_encodings.npy'), allow_pickle=True)
        known_face_names = np.load(os.path.join(known_faces_dir, 'known_face_names.npy'), allow_pickle=True)
        print("Face database loaded.")
    except FileNotFoundError:
        known_face_encodings = []
        known_face_names = []
        print("No saved face database found. Starting fresh.")
    return known_face_encodings, known_face_names

# Save the face database
def save_face_database(known_face_encodings, known_face_names):
    np.save(os.path.join(known_faces_dir, 'known_face_encodings.npy'), known_face_encodings)
    np.save(os.path.join(known_faces_dir, 'known_face_names.npy'), known_face_names)
    print("Face database saved.")

# Create a simple neural network for face recognition (adjust as necessary)
def create_face_recognition_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(160, 160, 3)),  # Input shape should match resized image shape
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Manually load and preprocess images from the training directory
def load_images_from_directory(directory):
    X = []
    y = []
    class_labels = os.listdir(directory)
    
    for label in class_labels:
        person_dir = os.path.join(directory, label)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                image_path = os.path.join(person_dir, filename)
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    image = cv2.imread(image_path)
                    image = cv2.resize(image, (160, 160))  # Resize images to fit the input size of the model
                    image = image / 255.0  # Normalize the pixel values
                    X.append(image)
                    y.append(class_labels.index(label))
    
    X = np.array(X)
    y = np.array(y)
    return X, y

# Train the model on manually loaded and processed images
def train_model(model, train_dir, epochs=10):
    # Load images and labels from training directory
    X, y = load_images_from_directory(train_dir)
    
    # Split data into train and validation sets (80% train, 20% validation)
    num_train = int(0.8 * len(X))
    X_train, X_val = X[:num_train], X[num_train:]
    y_train, y_val = y[:num_train], y[num_train:]

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))
    print("Model training completed.")

# Load known faces
known_face_encodings, known_face_names = load_face_database()

# Create or load the neural network model
model = create_face_recognition_model(len(known_face_names))

# Start video capture
video_capture = cv2.VideoCapture(0)

# Directory for saving images
saved_frames_dir = "saved_frames"
if not os.path.exists(saved_frames_dir):
    os.makedirs(saved_frames_dir)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Face detection
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Recognize faces
    for face_encoding, face_location in zip(face_encodings, face_locations):
        face_encoding_reshaped = np.expand_dims(face_encoding, axis=0)
        predictions = model.predict(face_encoding_reshaped, verbose=0)
        best_match_index = np.argmax(predictions)
        confidence = predictions[0][best_match_index]

        name = "Unknown"
        if confidence > 0.7:
            name = known_face_names[best_match_index]

        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({confidence:.2f})", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # If face is unknown, ask for the person's name and add to database
        if name == "Unknown":
            # Prompt to add new face to the database
            name = input("Enter the name of the new person: ")
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
            save_face_database(known_face_encodings, known_face_names)  # Save updated database

            # Save frame when face is unknown
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(saved_frames_dir, f"new_face_{name}_{timestamp}.jpg")
            cv2.imwrite(filename, frame)

            # Add image to training data directory for future training
            person_dir = os.path.join(train_dir, name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
            cv2.imwrite(os.path.join(person_dir, f"{timestamp}.jpg"), frame)

    # Train the model if needed (e.g., after adding new faces)
    model = train_model(model, train_dir)

    # Display the video  
    cv2.imshow('Security Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
