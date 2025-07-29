import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_DIR    = "final_model.h5"  # path to the SavedModel directory
DATA_DIR     = "./data"                   
CAM_INDEX    = 0
DETECTION_CONFIDENCE = 0.8
PADDING      = 150                       

# --- Dynamically load class names from data folder ---
classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
class_map = {idx: name for idx, name in enumerate(classes)}
print(f"Loaded {len(classes)} classes: {classes}")

# --- Load the trained model ---
model = load_model(MODEL_DIR)
print("Model loaded from", MODEL_DIR)

# --- MediaPipe Hands setup ---
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=DETECTION_CONFIDENCE
)

def extract_landmarks(rgb_img):
    """Run MediaPipe and return flattened 63‑vector or None."""
    res = hands.process(rgb_img)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0].landmark
    return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()

def standardize_landmarks(vec):
    """Normalize x and y to [0,1] based on min/max in the vector."""
    pts = vec.reshape(-1, 3)
    # x
    mn, mx = pts[:,0].min(), pts[:,0].max()
    pts[:,0] = (pts[:,0] - mn) / (mx - mn + 1e-8)
    # y
    mn, mx = pts[:,1].min(), pts[:,1].max()
    pts[:,1] = (pts[:,1] - mn) / (mx - mn + 1e-8)
    return pts.flatten()

# --- Open webcam ---
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # try normal detection
    lm_vec = extract_landmarks(rgb)

    # if no detection, pad and retry
    if lm_vec is None:
        padded = cv2.copyMakeBorder(
            frame, PADDING, PADDING, PADDING, PADDING,
            borderType=cv2.BORDER_CONSTANT, value=[0,0,0]
        )
        rgb_pad = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        lm_vec  = extract_landmarks(rgb_pad)
        if lm_vec is None:
            label = "No hand"
            cv2.putText(frame, label, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

    # standardize exactly as during training
    lm_vec = standardize_landmarks(lm_vec)

    # predict
    x_input    = lm_vec.reshape(1, -1)
    pred_probs = model.predict(x_input, verbose=0)[0]
    class_id   = int(np.argmax(pred_probs))
    label      = class_map[class_id]

    # draw landmarks
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
        )

    # overlay prediction
    cv2.putText(
        frame, f"Prediction: {label}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (255, 255, 255), 2, cv2.LINE_AA
    )

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
