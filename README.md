# 🦅 Edge-AI Animal Detection System
**Real-Time Computer Vision on Embedded Hardware using Quantized MobileNetV3**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red)
![Model](https://img.shields.io/badge/Model-MobileNetV3-green)
![Optimization](https://img.shields.io/badge/Optimization-ONNX%20INT8-orange)

## 📖 Project Overview
This project implements an autonomous **Edge-to-Cloud monitoring system** designed to detect animals in agricultural environments. 

Traditional computer vision systems often rely on heavy cloud processing, which requires high bandwidth and introduces latency. This solution runs **entirely on the edge** (Raspberry Pi 3 Model B), utilizing deep learning optimizations to classify video frames in real-time. Only critical detection events are synchronized to a **Firebase Realtime Database** for remote monitoring.

### Key Features
* **Edge Computing:** No internet required for detection; inference happens locally.
* **Model Optimization:** PyTorch model converted to **ONNX** and quantized to **INT8**, reducing model size by **~75%** and increasing inference speed by **3x**.
* **Cloud Sync:** Asynchronous event logging to Firebase (Label, Confidence, Timestamp).
* **Hardware Efficient:** Runs on low-power ARM CPUs (Raspberry Pi) without dedicated accelerators.

---
## 📊 Dataset
The model was trained on the **Farm Harmful Animals Dataset**, a collection of images specifically curated for agricultural monitoring.

* **Source:** [Kaggle - Farm Harmful Animals Dataset](https://www.kaggle.com/datasets/muzammilaliveltech/farm-harmful-animals-dataset)
* **Classes Detected:** The system is trained to identify specific species including **Wild Boar, Birds, Deer, Bear, Rabbit,** and others.
* **Preprocessing:** All images are resized to `128x128` and normalized before inference to match the input requirements of the MobileNetV3 backbone.

---


## 🏗️ System Architecture

The system follows a **Producer-Consumer** pattern:

1.  **Input:** The **CSI Camera** captures raw video frames (Producer).
2.  **Preprocessing:** Frames are resized to `128x128` and normalized.
3.  **Inference Engine:** `ONNX Runtime` executes the quantized **MobileNetV3** model.
4.  **Logic:** * If `Confidence > 70%`: Draw bounding box/label on local display.
    * If `Confidence > 70%` AND `Time_Since_Last_Alert > 5s`: Push data to Cloud.
5.  **Output:** **Firebase Realtime Database** receives the alert (Consumer).

---

## 🧠 Model & Optimization
To achieve real-time performance on a Raspberry Pi 3 (which has limited CPU power), standard models like ResNet or VGG are too heavy. We utilized **MobileNetV3-Large** and applied post-training dynamic quantization.

### 1. The Backbone
We selected **MobileNetV3** due to its lightweight "Inverted Residual" blocks and "Squeeze-and-Excitation" modules, which are specifically designed for mobile CPUs.

### 2. Quantization Process (FP32 vs INT8)
Deep learning models typically train using 32-bit floating-point numbers (FP32). By mapping these values to 8-bit integers (INT8), we significantly reduce the memory bandwidth required.

| Metric | Original (FP32) | Optimized (INT8) | Improvement |
| :--- | :--- | :--- | :--- |
| **Model Size** | ~15.2 MB | ~3.9 MB | **74% Smaller** |
| **Inference Time (Pi 3)** | ~350 ms | ~90-110 ms | **~3.5x Faster** |
| **Accuracy Loss** | Baseline | < 1.5% drop | Negligible |

*The quantization script is included in `optimizer.py`.*

---

## 📂 Repository Structure

```text
├── core/
│   └── inference.py                    # Main execution script for the Raspberry Pi
│   └── optimizer.py                    # Script to convert PyTorch .pth -> ONNX INT8
├── models/
│   ├── baseline_fp32.onnx              # Standard model (Reference)
│   └── optimized_int8.onnx             # Quantized model (Production)
├── config/
│   └── class_names.json                # List of animal classes (Boar, Deer, Crow, etc.)
├── assets/
│   └── demo_video.mp4                  # System demonstration
│   └── system_architecture.jpg         # System demonstration
├── requirements.txt                    # Dependencies
└── README.md                           # Documentation