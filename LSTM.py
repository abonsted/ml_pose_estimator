import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import numpy as np

classes = [
    "baseball_pitch",
    "baseball_swing",
    "bench_press",
    "bowl",
    "clean_and_jerk",
    "golf_swing",
    "jump_rope",
    "jumping_jacks",
    "pullup",
    "pushup",
    "situp",
    "squat",
    "strum_guitar",
    "tennis_forehand",
    "tennis_serve",
]

# Function to extract features for a single video
def extract_video_features(frames_dir, max_frames, transform, cnn_model):
    print("Loading " + frames_dir)
    frames = []
    for file_name in sorted(os.listdir(frames_dir)):
        img = Image.open(os.path.join(frames_dir, file_name)).convert("L")
        frames.append(transform(img))
    
    # Truncate or pad frames
    if len(frames) > max_frames:
        frames = frames[:max_frames]
    elif len(frames) < max_frames:
        padding = [torch.zeros_like(frames[0]) for _ in range(max_frames - len(frames))]
        frames.extend(padding)
    
    # Stack frames and extract CNN features
    frames = torch.stack(frames)  # Shape: (max_frames, C, H, W)
    frames = frames.view(max_frames, *frames[0].size())  # Ensure proper shape
    with torch.no_grad():
        features = cnn_model(frames)  # Shape: (max_frames, cnn_output_size)

    features = features.view(features.size(0), -1)
    return features

videos_dir = "Penn_Action/Penn_Action/frames"
labels_dir = "Penn_Action/Penn_Action/labels"
max_frames = 70
img_height, img_width = 64, 64
num_classes = 15

# Pre-trained CNN for feature extraction
cnn_model = models.resnet18()
cnn_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
cnn_model = nn.Sequential(*list(cnn_model.children())[:-1])  # Remove classification head
cnn_model.eval()  # Freeze CNN during training

# Data transforms
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Prepare data
X = []
y = []

test = 0
for video_folder in sorted(os.listdir(videos_dir)):
    if video_folder == ".DS_Store":
        continue

    video_frames_path = os.path.join(videos_dir, video_folder)
    matlab_file_path = os.path.join(labels_dir, f"{video_folder}.mat")
    
    # Extract features for video
    video_features = extract_video_features(video_frames_path, max_frames, transform, cnn_model)
    
    # Load label
    mat = loadmat(matlab_file_path)
    video_label = mat['action'].item()  # Assuming label is scalar
    
    X.append(video_features)
    y.append(classes.index(video_label))
    test += 1

# Convert to tensors
X = torch.stack(X)  # Shape: (n_videos, max_frames, cnn_output_size)
y = torch.tensor(y)  # Shape: (n_videos,)

# Split data into train and test sets
train_indices, test_indices = train_test_split(range(len(X)), test_size=0.2, random_state=42)
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

class VideoClassifierLSTM(nn.Module):
    def __init__(self, lstm_hidden_size, num_classes, input_size):
        super(VideoClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # Shape of hidden: (1, batch_size, lstm_hidden_size)
        output = self.fc(hidden[-1])  # Shape: (batch_size, num_classes)
        return output

# Model setup
lstm_hidden_size = 128
cnn_output_size = 512  # For ResNet18; adjust if using another model
model = VideoClassifierLSTM(lstm_hidden_size, num_classes, input_size=cnn_output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
batch_size = 4

for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")