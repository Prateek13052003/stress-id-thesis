# %%
import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('my_test_video.mp4', fourcc, 30.0, (640, 480))

print("Recording... Press 'q' to stop")
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

# %%
!pip install opencv-python

# %%
import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('my_test_video.mp4', fourcc, 30.0, (640, 480))

print("Recording... Press 'q' to stop")
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

# %%
import cv2

cap = cv2.VideoCapture(0)

# Check if camera opened
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Get actual frame dimensions from camera
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Use XVID codec with .avi — most compatible
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('my_test_video.avi', fourcc, 20.0, (width, height))

if not out.isOpened():
    print("VideoWriter failed to open — trying fallback codec")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('my_test_video.avi', fourcc, 20.0, (width, height))

print("Recording... Press 'q' to stop")
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('Recording - Press Q to stop', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to grab frame")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved as my_test_video.avi")

# %%
import os

# Check what's available in Feature Extraction
for root, dirs, files in os.walk('../Feature Extraction'):
    for file in files:
        print(os.path.join(root, file))

# %%
import pandas as pd

# Check the video feature files the model was trained on
df = pd.read_csv('../Feature Extraction/Features/video7tasks_aus_gaze_mean_std.csv')
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(2))

# %%
with open('../Feature Extraction/README.md', 'r') as f:
    print(f.read())

# %%
import pandas as pd

df = pd.read_csv('../Feature Extraction/Features/video7tasks_aus_gaze_mean_std.csv')
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# %%
!pip install feat
!pip install torch torchvision  # dependency for py-feat

# %%
!pip uninstall feat -y
!pip install py-feat==0.5.0

# %% [markdown]
# 

# %%
from feat import Detector
import pandas as pd

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model="xgb",
    emotion_model="resmasknet",
    facepose_model="img2pose"
)

video_path = "my_test_video.avi"

# Process with more skip frames and error handling
try:
    video_predictions = detector.detect_video(
        video_path, 
        skip_frames=10,        # skip more frames to avoid bad ones
        batch_size=1,          # process one frame at a time
        num_workers=0          # avoid multiprocessing issues on Mac
    )
    print("Success!")
    print(video_predictions.shape)
    print(video_predictions.head())
except Exception as e:
    print(f"Error: {e}")
    # Fallback: process frame by frame manually
    import cv2
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % 15 == 0:  # every 15th frame
            try:
                cv2.imwrite('temp_frame.jpg', frame)
                pred = detector.detect_image('temp_frame.jpg')
                if pred is not None and len(pred) > 0:
                    results.append(pred)
            except:
                pass
        frame_num += 1
    cap.release()
    
    if results:
        video_predictions = pd.concat(results, ignore_index=True)
        print("Fallback success! Shape:", video_predictions.shape)
    else:
        print("No faces detected at all — check lighting/camera angle")

# %%
import pandas as pd
import numpy as np

# Check what columns we have
au_cols = [c for c in video_predictions.columns if 'AU' in c]
gaze_cols = [c for c in video_predictions.columns if 'gaze' in c or 'pose' in c]

print("AU columns:", au_cols)
print("Gaze columns:", gaze_cols)

# %%
# Compute mean and std to match training format
feature_cols = au_cols + gaze_cols

mean_feats = video_predictions[feature_cols].mean()
std_feats = video_predictions[feature_cols].std()

# Rename to match training CSV format
mean_feats.index = ['mean_' + c for c in feature_cols]
std_feats.index = ['std_' + c for c in feature_cols]

# Combine into one row
my_features = pd.concat([mean_feats, std_feats]).to_frame().T
my_features.insert(0, 'id', 'my_test')

print("My feature shape:", my_features.shape)
print(my_features.columns.tolist())

# %%
# Now compare with training columns
train_df = pd.read_csv('../Feature Extraction/Features/video7tasks_aus_gaze_mean_std.csv')
train_cols = train_df.columns.tolist()

my_cols = my_features.columns.tolist()

print("Missing from my features:", [c for c in train_cols if c not in my_cols])
print("Extra in my features:", [c for c in my_cols if c not in train_cols])

# %%
# py-feat AU values are intensity (like _r), we can map them
# For _c (presence), we threshold: AU > 0.5 means present (1), else 0

au_list = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10',
           'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25', 'AU26', 'AU45']

# Map AU45 from AU43 (py-feat uses AU43 for blink, OpenFace uses AU45)
video_predictions['AU45'] = video_predictions.get('AU43', 0)

# Build the final feature row
final_features = {'id': 'my_test'}

for au in au_list:
    col = au  # py-feat column name
    if col in video_predictions.columns:
        vals = video_predictions[col]
        final_features[f'mean_{au}_r'] = vals.mean()
        final_features[f'std_{au}_r']  = vals.std()
        final_features[f'mean_{au}_c'] = (vals > 0.5).astype(int).mean()
        final_features[f'std_{au}_c']  = (vals > 0.5).astype(int).std()
    else:
        final_features[f'mean_{au}_r'] = 0
        final_features[f'std_{au}_r']  = 0
        final_features[f'mean_{au}_c'] = 0
        final_features[f'std_{au}_c']  = 0

# Map gaze columns
gaze_map = {
    'mean_gaze_0_x': 'x_0', 'mean_gaze_0_y': 'y_0', 'mean_gaze_0_z': None,
    'mean_gaze_1_x': 'x_1', 'mean_gaze_1_y': 'y_1', 'mean_gaze_1_z': None,
    'mean_gaze_angle_x': None, 'mean_gaze_angle_y': None,
}

# Check available gaze cols in our data
print("Available gaze cols:", [c for c in video_predictions.columns if 'gaze' in c.lower() or 'Gaze' in c])
print("Available pose cols:", [c for c in video_predictions.columns if 'pose' in c.lower()])

# %%
# After seeing gaze cols above, map them manually
# Fill gaze with 0 for now and we'll fix after seeing column names
for col in ['mean_gaze_0_x','mean_gaze_0_y','mean_gaze_0_z',
            'mean_gaze_1_x','mean_gaze_1_y','mean_gaze_1_z',
            'mean_gaze_angle_x','mean_gaze_angle_y',
            'std_gaze_0_x','std_gaze_0_y','std_gaze_0_z',
            'std_gaze_1_x','std_gaze_1_y','std_gaze_1_z',
            'std_gaze_angle_x','std_gaze_angle_y']:
    final_features[col] = 0  # placeholder — will fix after seeing gaze cols

my_features_df = pd.DataFrame([final_features])

# Reorder to match training columns exactly
train_df = pd.read_csv('../Feature Extraction/Features/video7tasks_aus_gaze_mean_std.csv')
my_features_df = my_features_df.reindex(columns=train_df.columns, fill_value=0)

print("Shape:", my_features_df.shape)
print("Missing cols:", my_features_df.isnull().sum().sum())
print(my_features_df.head())

# %%
# Check what classifiers are available in the Results folder
import os
for root, dirs, files in os.walk('../Classification/Results'):
    for file in files:
        print(os.path.join(root, file))

# %%
import pickle
import pandas as pd
import numpy as np

# Prepare input (drop id column)
X_test = my_features_df.drop(columns=['id'])

# Load the trained model — check the Results folder path from above
# Try loading the video classifier
model_path = '../Classification/Results/your_model.pkl'  # update path after running cell above

model = pickle.load(open(model_path, 'rb'))

# Predict
prediction = model.predict(X_test)
proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

print("Predicted class:", prediction[0])
if proba is not None:
    print("Confidence scores:", proba)

# Map label to meaning
label_map = {0: 'No Stress / Low Stress', 1: 'Stress / Depressed', 2: 'High Stress'}
print("Result:", label_map.get(prediction[0], prediction[0]))

# %%
import os

# List everything in the Results folder
for root, dirs, files in os.walk('../Classification/Results'):
    for file in files:
        print(os.path.join(root, file))

# %%
# Read the classification video notebook to understand the model
import json

with open('../Classification/classification_video.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'classifier' in src.lower() or 'fit' in src.lower() or 'list_classif' in src.lower():
            print("=== CELL ===")
            print(src)
            print()

# %%
import sys
sys.path.append('../Classification')

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # must come before this
from sklearn.impute import IterativeImputer

print("All imports successful!")

# %%
import json

with open('../Classification/classification_video.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = ''.join(cell['source'])
    if 'labels' in src.lower() and 'read_csv' in src:
        print("=== CELL ===")
        print(src)
        print()

# %%
import sys
sys.path.append('../Classification')

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load training data and labels
train_df = pd.read_csv('../Feature Extraction/Features/video7tasks_aus_gaze_mean_std.csv')
labels = pd.read_csv('../../Dataset/labels.csv', sep=",", header=0, index_col=0).dropna()

print("Train shape:", train_df.shape)
print("Labels shape:", labels.shape)
print("Label columns:", labels.columns.tolist())
print(labels.head())

# %%
import os

# Search for labels.csv anywhere on the project
for root, dirs, files in os.walk('../../'):
    for file in files:
        if 'label' in file.lower():
            print(os.path.join(root, file))

# %%
# Search more broadly for any csv with 'affect' or 'stress' in it
import os

for root, dirs, files in os.walk('../../stressID-main/'):
    for file in files:
        print(os.path.join(root, file))

# %%
# Also check if labels are inside the Classification folder
for root, dirs, files in os.walk('../'):
    for file in files:
        if file.endswith('.csv'):
            print(os.path.join(root, file))

# %%
import json

with open('../../stressID-main/Labels_preparation.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = ''.join(cell['source'])
    if src.strip():
        print("=== CELL ===")
        print(src)
        print()

# %%
import pandas as pd

train_df = pd.read_csv('../Feature Extraction/Features/video7tasks_aus_gaze_mean_std.csv')
my_df = my_features_df.drop(columns=['id'])

print("=== Training data stats ===")
print(train_df.drop(columns=['id']).describe().loc[['mean', 'std']].T.head(10))

print("\n=== Your video stats ===")
print(my_df.describe().loc[['mean', 'std']].T.head(10))

# %%
import pandas as pd

train_df = pd.read_csv('../Feature Extraction/Features/video7tasks_aus_gaze_mean_std.csv')

print("Training data shape:", train_df.shape)
print("My extracted features shape:", my_features_df.shape)
print("Columns match:", list(train_df.columns) == list(my_features_df.columns))
print("\nMy extracted features:")
print(my_features_df.T)

# %%



