import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd

# --- CONFIGURATION ---
DATA_DIR     = "./data"             # root folder with subfolders A, B, ..., Y (no J,Z)
IMAGE_SIZE   = (224, 224)           # resize all images
PADDING      = 150                  # border size when retrying detection
MIN_CONF     = 0.5                  # MediaPipe min_detection_confidence

# --- INITIALIZE MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=MIN_CONF
)

# --- DISCOVER CLASSES DYNAMICALLY ---
classes    = sorted([d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d))])
class_map  = {cls_name: idx for idx, cls_name in enumerate(classes)}
print(f"Found {len(classes)} classes: {classes}")

# --- HELPERS ---
def extract_landmarks(img_rgb):
    """Run MediaPipe, return flattened 63‑vector or None."""
    res = hands.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0].landmark
    return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()

def standardize_landmarks(vec):
    """Reshape to (21,3), normalize x and y independently to [0,1], then flatten back."""
    pts = vec.reshape(-1, 3)
    # x coords
    min_x, max_x = pts[:,0].min(), pts[:,0].max()
    pts[:,0] = (pts[:,0] - min_x) / (max_x - min_x + 1e-8)
    # y coords
    min_y, max_y = pts[:,1].min(), pts[:,1].max()
    pts[:,1] = (pts[:,1] - min_y) / (max_y - min_y + 1e-8)
    return pts.flatten()

def process_image(img):
    """
    Given a BGR image, resize, pad/retry if needed, extract standardized landmarks.
    Returns a list of landmark vectors (could be empty or length=1).
    """
    # resize + convert
    img = cv2.resize(img, IMAGE_SIZE)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # try normally
    lm = extract_landmarks(rgb)

    # if fail, pad & retry
    if lm is None:
        padded = cv2.copyMakeBorder(
            img, PADDING, PADDING, PADDING, PADDING,
            borderType=cv2.BORDER_CONSTANT, value=[0,0,0]
        )
        rgb_pad = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        lm = extract_landmarks(rgb_pad)
        if lm is None:
            return []

    # standardize
    lm = standardize_landmarks(lm)
    return [lm]

# --- DATA COLLECTION WITH FLIP AUGMENTATION ---
all_data   = []
all_labels = []

for cls_name, cls_idx in class_map.items():
    folder = os.path.join(DATA_DIR, cls_name)
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img  = cv2.imread(path)
        if img is None:
            continue

        # process original
        vecs = process_image(img)
        for v in vecs:
            all_data.append(v)
            all_labels.append(cls_idx)

        # process flipped
        img_flip = cv2.flip(img, 1)  # horizontal flip
        vecs_flip = process_image(img_flip)
        for v in vecs_flip:
            all_data.append(v)
            all_labels.append(cls_idx)

hands.close()

# --- CONVERT TO ARRAYS ---
X = np.stack(all_data).astype(np.float32)
y = np.array(all_labels, dtype=np.int32)
print(f"Collected {len(y)} samples across {len(classes)} classes.")

# --- SAVE PICKLE ---
with open("asl_data_augmented.pickle", "wb") as f:
    pickle.dump({"data": X, "labels": y, "classes": classes}, f)
print("Saved asl_data_augmented.pickle")

# --- SAVE CSV ---
cols = []
for i in range(21):
    cols += [f"x{i}", f"y{i}", f"z{i}"]
df = pd.DataFrame(X, columns=cols)
df["label"] = y
df["class_name"] = df["label"].map({v:k for k,v in class_map.items()})
df.to_csv("asl_data_augmented.csv", index=False)
print("Saved asl_data_augmented.csv")
