#Andrew Bonsted
# The following program is inspired by the following github, adapted to our use-case
# https://github.com/computervisioneng/image-classification-python-scikit-learn/blob/master/main.py

#This model is NOT the one we chosen to pursue for our midterm report
    #It has never been able to finish running with how much data there is

import os
import pickle
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Lists used to hold pictures and cooresponding labels
unique_video_nums = []
unique_labels = []

video_nums = []
data = []
labels = []

# Directory path to frames and labels
frames_dir = '.\\Penn_Action\\Penn_Action\\frames'
labels_dir = '.\\Penn_Action\\Penn_Action\\labels'

# Iterate through all data records and frames
for num, folder in enumerate(os.listdir(frames_dir)):
    # Keeping track of overall video number and cooresponding label
    print(folder)
    label_path = os.path.join(labels_dir, folder + ".mat")
    mat = scio.loadmat(label_path)
    unique_labels.append(mat["action"])
    unique_video_nums.append(folder)

    # Iterating through all of the frames in a video
    for file in os.listdir(os.path.join(frames_dir, folder)):
        video_nums.append(folder)

        # Build img/label path
        img_path = os.path.join(frames_dir, folder, file)

        # Retrieving image from path, resizing, and adding to data
        img = imread(img_path)
        img = resize(img, (20, 20))
        data.append(img.flatten())

        # Printing resized image
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()

        labels.append(mat["action"][0])
    #     break
    # if num == 200:
    #     break

# Splitting the videos into test and train sets
train_video_nums, test_video_nums = train_test_split(unique_video_nums, test_size=0.1, random_state=42, stratify=unique_labels)

# Iterating through all of data to separate them according to the split above
train_frames = []
train_labels = []
test_frames = []
test_labels = []
for i in range(len(data)):
    if video_nums[i] in train_video_nums:
        train_frames.append(data[i])
        train_labels.append(labels[i])
    elif video_nums[i] in test_video_nums:
        test_frames.append(data[i])
        test_labels.append(labels[i])

# print(train_video_nums)
# print(test_video_nums)
# print(len(train_labels))
# print(len(test_labels))

# Training Support Vector Classifier
    #This is the step that has never finished before and has caused us to find a new model
classifier = SVC()
parameters = [{'gamma': [0.01], 'C': [10]}] # [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(train_frames, train_labels)

#Best estimator
print("best estimator")
best_estimator = grid_search.best_estimator_

#Predict based on test frames
y_pred = best_estimator.predict(test_frames)

#Calculate and display accuracy score
score = accuracy_score(y_pred, test_labels)
print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))