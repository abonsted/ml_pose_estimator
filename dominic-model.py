import scipy
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import random

# Function will load the matlab file from the given file path and extract the "action" label.
def load_labels(file_path):
    mat = scipy.io.loadmat(file_path)
    labels = mat['action'].item()
    return labels

# Function will load the frames from the given directory and stop if it reaches the target # of frames.
# The motivation for the target number of frames is that each video will be a sample for our model, but
# each video has a different number of frames which creates an inhomogeneous array.
def load_frames(frames_dir, target_num_frames):
    print("Loading " + frames_dir)
    frames = []
    for file in sorted(os.listdir(frames_dir)):
        img = cv2.imread(os.path.join(frames_dir, file), cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (64, 64))
        frames.append(img_resized.flatten())
        
        if len(frames) == target_num_frames:
            break

    while len(frames) < target_num_frames:
        frames.append(frames[random.randint(0, len(frames)-1)])

    return np.concatenate(frames)

target_num_frames = 50
data = []
labels = []

videos_dir = "Penn_Action/Penn_Action/frames"
labels_dir = "Penn_Action/Penn_Action/labels"

# For each video directory in our dataset, extract said directory and corresponding matlab file
for video_folder in sorted(os.listdir(videos_dir)):
    if video_folder == ".DS_Store":
        continue
    video_frames_path = os.path.join(videos_dir, video_folder)
    matlab_file_path = os.path.join(labels_dir, f"{video_folder}.mat")

    video_features = load_frames(video_frames_path, target_num_frames)
    label = load_labels(matlab_file_path)

    data.append(video_features)
    labels.append(label)

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

report = classification_report(y_test, y_pred)
print(report)