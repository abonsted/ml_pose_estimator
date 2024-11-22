import scipy
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import random
from sklearn.tree import DecisionTreeClassifier, plot_tree
from scipy.stats import t
from sklearn.decomposition import PCA

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

# def evaluate(X_train, y_train, folds):
#     # depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
#     estimators = [100, 200, 300, 400, 500]
#     train_scores = []
#     cv_means = []
#     cv_stds = []

#     for e in estimators:
#         print(f"Evaluating n_estimators={e}")
#         clf = RandomForestClassifier(n_estimators=e, random_state=42)

#         clf.fit(X_train, y_train)
#         train_scores.append(clf.score(X_train, y_train))

#         cv_scores = cross_val_score(clf, X_train, y_train, cv=folds)
#         cv_means.append(np.mean(cv_scores))
#         cv_stds.append(np.std(cv_scores))

#     cv_means = np.array(cv_means)
#     cv_stds = np.array(cv_stds)

#     plt.figure(figsize = (8, 6))
#     plt.plot(estimators, train_scores, label = "Training set scores", marker = "o")
#     plt.errorbar(estimators, cv_means, yerr = cv_stds, fmt = "-o", capsize = 5, label = "Mean CV scores (plus/minus errors)", color = "orange")
#     plt.fill_between(estimators, 
#                     cv_means + cv_stds, 
#                     cv_means - cv_stds, 
#                     linestyle = "--",
#                     alpha = 0.1, color = "orange")

#     plt.xlabel("Number of Estimators)")
#     plt.ylabel("Accuracy")
#     plt.title("Decision Tree Performance")
#     plt.legend()
#     plt.show()

def k_fold_validation(data, labels, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        pca = PCA(n_components=50)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_pca, y_train)
        
        y_pred = model.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        report = classification_report(y_test, y_pred, zero_division=0)
        print(report)
        
        cv_score = cross_val_score(model, X_train_pca, y_train, cv=k)
        cv_scores.append(cv_score)
    
    return np.array(cv_scores)

def confidence(cv_scores, confidence=0.95):
    n = len(cv_scores)
    mean = np.mean(cv_scores)
    std_err = np.std(cv_scores) / np.sqrt(n)
    t_critical = t.ppf((1 + confidence) / 2, df=n-1)
    h = t_critical * std_err
    return mean, mean - h, mean + h

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

cv_scores = k_fold_validation(X, y, 5)
mean, ci_lower, ci_upper = confidence(cv_scores, confidence=0.95)
print(f'Mean CV score = {mean: .2f}')
print(f'Confidence interval: ({ci_lower: .2f}, {ci_upper: .2f})')

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # evaluate(X_train, y_train, 5)

# clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# conf_mat = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(10, 7))
# sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()

# report = classification_report(y_test, y_pred, zero_division=0)
# print(report)