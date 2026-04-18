#  Real-Time Indian Sign Language Detection System
Real-time ISL alphabet recognition using webcam, MediaPipe landmarks (126 features), and ANN model. Achieves 20-30 FPS on standard laptops.

# Features:
-Static ISL alphabet (A-Z) recognition
-Two-hand gesture support
-Webcam-based, no sensors required
-Real-time prediction with smoothing
-Open-source stack (Python + MediaPipe + TensorFlow)

# Dependencies:
-Python
-Opencv 
-Mediapipe 
-Tensorflow 
-Numpy & pandas

Position hand 30cm from webcam. System displays recognized ISL letter live.

# Results
-Accuracy: High training/validation
-Speed: 20-30 FPS real-time
-Dataset: 200 samples/gesture, 36 classes (A-Z, 0-9)

# Workkflow:
- Webcam → OpenCV → MediaPipe (landmarks) → ANN → Text Output
