#!/usr/bin/env python3
import time
import json
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
from picamera2 import Picamera2
import onnxruntime as ort

import firebase_admin
from firebase_admin import credentials, db


MODEL_PATH = "mobilenetv3_pest_fp32.onnx"
CLASS_NAMES_PATH = "class_names.json"

INPUT_SIZE = 128  # training size (128x128)

# --- Firebase config ---
# Path to your service account JSON on the Pi
FIREBASE_CRED_PATH = "/home/mohit/AML_project/Firebase_auth.json"

# Your Realtime Database URL from Firebase console
# Change this if your URL is slightly different
FIREBASE_DB_URL = "https://farmalert-7eb58-default-rtdb.europe-west1.firebasedatabase.app/"

# Minimum probability to send an event to Firebase
PEST_CONF_THRESHOLD = 0.70

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)
# ======================
# FIREBASE SETUP
# ======================

def init_firebase():
    """Initialize Firebase app and get a reference to /detections."""
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred, {
            "databaseURL": FIREBASE_DB_URL
        })
    return db.reference("detections")


FIREBASE_DETECTIONS_REF = init_firebase()


def send_detection_to_firebase(label: str, prob: float, latency_ms: float):
    """Push one detection event to Firebase Realtime Database."""
    data = {
        "label": label,
        "probability": float(prob),
        "latency_ms": float(latency_ms),
        "is_pest": label in CLASS_NAMES,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    try:
        FIREBASE_DETECTIONS_REF.push(data)
        print("Sent to Firebase:", data)
    except Exception as e:
        print("Firebase error:", e)


# ======================
# MODEL / PREPROCESSING
# ======================




def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    frame: HxWx3 RGB uint8 from Picamera2
    returns: 1x3xHxW float32 tensor with ImageNet-style normalization
    """
    img = Image.fromarray(frame).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

    arr = np.asarray(img).astype("float32") / 255.0  # HWC in [0,1]
    arr = arr.transpose(2, 0, 1)  # CHW

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    arr = (arr - mean) / std

    arr = np.expand_dims(arr, axis=0)  # NCHW
    return arr.astype("float32")


def load_onnx_session():
    """Create ONNX Runtime session on CPU."""
    sess = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    return sess, input_name


# ======================
# MAIN LOOP
# ======================

def main():
    # ONNX Runtime
    sess = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
        
    )
    input_name = sess.get_inputs()[0].name

    # Camera setup
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # warm-up

    print("Camera running. A window will open with live feed.")
    print("Press 'q' in the window to quit.")

    last_sent_time = 0.0  # to throttle Firebase events

    try:
        while True:
            # Get RGB frame from camera
            frame_rgb = picam2.capture_array()

            # Prepare input tensor
            x = preprocess(frame_rgb)

            # Inference
            t0 = time.time()
            logits, = sess.run(None, {input_name: x})
            dt_ms = (time.time() - t0) * 1000.0

            # Softmax
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = exp / exp.sum(axis=1, keepdims=True)

            pred_idx = int(np.argmax(probs, axis=1)[0])
            pred_prob = float(probs[0, pred_idx])
            label = CLASS_NAMES[pred_idx]

            # Convert RGB -> BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Text overlay
            text = f"{label}  {pred_prob*100:.1f}%  ({dt_ms:.1f} ms)"
            cv2.putText(
                frame_bgr,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Pest classifier", frame_bgr)

            # Print to terminal as well
            print(f"[{dt_ms:6.1f} ms] {label:20s}  {pred_prob*100:5.1f}%")

            # Send to Firebase if it's a pest and above threshold,
            # and not more than once per 5 seconds
            now = time.time()
            if (
                label in CLASS_NAMES
                and pred_prob >= PEST_CONF_THRESHOLD
                and now - last_sent_time > 5.0
            ):
                send_detection_to_firebase(label, pred_prob, dt_ms)
                last_sent_time = now

            # Handle keypress
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
