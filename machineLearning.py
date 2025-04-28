# ml_classifier.py
import numpy as np
from sklearn.svm import SVC

# Load softmax outputs and true labels saved from the OpenCV + Neural Network
softmax_outputs = np.load('softmax_outputs.npy')
true_labels = np.load('true_labels.npy', allow_pickle=True)

# Train SVM on the softmax outputs
clf = SVC(kernel='linear', probability=True)
clf.fit(softmax_outputs, true_labels)

print("✅ SVM trained on Neural Network outputs.")

import joblib
joblib.dump(clf, 'svm_face_classifier.joblib')

# Example of how to predict (later during live use)
def predict_with_svm(softmax_vector):
    softmax_vector = np.array(softmax_vector).reshape(1, -1)  
    prediction = clf.predict(softmax_vector)[0]
    confidence = clf.predict_proba(softmax_vector)[0].max()
    return prediction, confidence
