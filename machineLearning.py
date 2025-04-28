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

