#Andrew Bonsted
#This program was made to create Table 1 in our Midterm Report
    #Collecting the statistics of: 1) Number of videos per exercise and 2) Average Frames per Video for each exercise

import pandas as pd
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import os
from collections import Counter

labels_dir = '.\\Penn_Action\\Penn_Action\\labels'
labels = []

# Retrieves each label and counts how many there are
for file in os.listdir(labels_dir):
    label_path = os.path.join(labels_dir, file)
    mat = scio.loadmat(label_path)

    labels.append(mat["action"][0])
    print(mat)
    break

labels_count = Counter(labels)

print(labels_count)


#Finds average number of frames per video for each exercise class
frames_dir = '.\\Penn_Action\\Penn_Action\\frames'
frame_count = 0

video_frames = []
for num, folder in enumerate(os.listdir(frames_dir)):
    folder_path = os.path.join(frames_dir, folder)

    files = os.listdir(folder_path)
    video_frames.append(len(files))

print(len(video_frames))
print(len(labels))
print(labels_count["pullup"])

for x in labels_count:
    count = 0
    for num, label in enumerate(labels):
        if label == x:
            count += video_frames[num]
    
    print(f"{x}: {count / labels_count[x]}")
