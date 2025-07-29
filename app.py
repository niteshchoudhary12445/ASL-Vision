import os
import time
import threading
import difflib

import cv2
import numpy as np
import mediapipe as mp

from flask import Flask, render_template, Response, jsonify, request
from tensorflow.keras.models import load_model

# -------- Configuration --------
CAM_SOURCE    = int(os.getenv("CAM_SOURCE", 0))
WIDTH, HEIGHT = 640, 480
FPS_TARGET    = 30
DETECTION_CONF = 0.8
SUGGESTION_LIMIT = 5

# -------- Threaded Capture Class --------
class ThreadedCamera:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False

        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self.stopped:
            if self.cap.grab():
                _, img = self.cap.retrieve()
                with self.lock:
                    self.frame = img
            else:
                time.sleep(0.001)

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        self.cap.release()

# -------- Global State --------
_last_probs = []
_history    = []
_frame      = None
_lock       = threading.Lock()
_current_str = ""
_suggestions = []

# -------- Load Classes & Model --------
DATA_DIR = "./data"
classes = sorted(d for d in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR, d)))
class_map = {i: c for i, c in enumerate(classes)}
try:
    model = load_model(os.getenv("MODEL_PATH", "final_model.h5"))
except Exception as e:
    print(f"[ERROR] Loading model failed: {e}")
    model = None

# -------- Load Word List for Suggestions --------
WORD_FILE = os.path.join(DATA_DIR, "words.txt")
if os.path.exists(WORD_FILE):
    with open(WORD_FILE) as f:
        WORDS = [w.strip().lower() for w in f if w.strip()]
else:
    # fallback to system words file
    try:
        with open("/usr/share/dict/words") as f:
            WORDS = [w.strip().lower() for w in f if w.strip()]
    except:
        WORDS = []  # no suggestions available

# -------- MediaPipe Setup --------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# -------- Start Camera Thread --------
cam = ThreadedCamera(src=CAM_SOURCE, width=WIDTH, height=HEIGHT)
time.sleep(1.0)

def capture_loop():
    global _frame, _last_probs, _history, DETECTION_CONF, _current_str, _suggestions
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=DETECTION_CONF,
        min_tracking_confidence=DETECTION_CONF
    )
    interval = 1.0 / FPS_TARGET
    counter  = 0
    prev_label = None

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.005)
            continue

        img = cv2.flip(frame, 1)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = "No hand"
        probs = [0.0] * len(classes)

        if counter % 2 == 0 and model:
            hands.min_detection_confidence = DETECTION_CONF
            hands.min_tracking_confidence  = DETECTION_CONF
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                pts = np.array(
                    [[p.x, p.y, p.z] for p in res.multi_hand_landmarks[0].landmark],
                    dtype=np.float32
                )
                for d in (0, 1):
                    mn, mx = pts[:, d].min(), pts[:, d].max()
                    pts[:, d] = (pts[:, d] - mn) / (mx - mn + 1e-8)

                x_in  = pts.flatten().reshape(1, -1)
                preds = model.predict(x_in, verbose=0)[0].tolist()
                idx   = int(np.argmax(preds))
                label = class_map.get(idx, label)
                probs = preds

                with _lock:
                    _last_probs = preds
                    _history.insert(0, label)
                    _history[:] = _history[:20]

                # Build current string and suggestions
                if label != prev_label and label.isalpha() and len(label) == 1:
                    _current_str += label.lower()
                    # prefix filter
                    _suggestions = [w for w in WORDS if w.startswith(_current_str)]
                    if len(_suggestions) > SUGGESTION_LIMIT:
                        _suggestions = _suggestions[:SUGGESTION_LIMIT]
                prev_label = label

                # Draw landmarks and label
                mp_draw.draw_landmarks(img, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(img, (5,5), (200,45), (0,0,0), -1)
                cv2.putText(img, label, (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        counter += 1
        with _lock:
            _frame = img.copy()

        time.sleep(interval)

threading.Thread(target=capture_loop, daemon=True).start()

# -------- Flask App --------
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', classes=classes, initial_conf=DETECTION_CONF)

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            with _lock:
                frame = _frame
            if frame is None:
                continue
            ret, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/probabilities')
def probabilities():
    with _lock:
        return jsonify(probs=_last_probs, history=_history)

@app.route('/suggestions')
def suggestions():
    with _lock:
        return jsonify(current=_current_str, suggestions=_suggestions)

@app.route('/set_sensitivity', methods=['POST'])
def set_sensitivity():
    global DETECTION_CONF
    try:
        v = float(request.json.get('value', DETECTION_CONF))
        DETECTION_CONF = max(0.0, min(1.0, v))
    except (ValueError, TypeError):
        return jsonify(error="Invalid sensitivity"), 400
    return ('', 204)

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, debug=False)