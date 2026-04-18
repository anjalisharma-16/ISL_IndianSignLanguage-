import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    
    base_x, base_y, base_z = landmarks[0], landmarks[1], landmarks[2]
    
    for i in range(0, len(landmarks), 3):
        landmarks[i] -= base_x
        landmarks[i+1] -= base_y
        landmarks[i+2] -= base_z
    
    max_value = max(abs(landmarks))
    if max_value != 0:
        landmarks = landmarks / max_value
    
    return landmarks

# Load model and labels with allow_pickle=True
model = load_model("model.h5")
labels = np.load("labels.npy", allow_pickle=True)  # Fixed here

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

prev_time = 0
history = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    text = "Show Both Hands"
    landmarks = []
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        
        # Check if we have exactly 2 hands (126 landmarks)
        if len(landmarks) == 126:
            landmarks = normalize_landmarks(landmarks)
            landmarks = np.array(landmarks).reshape(1, -1)
            
            pred = model.predict(landmarks, verbose=0)
            idx = np.argmax(pred)
            label = labels[idx]
            
            history.append(label)
            if len(history) > 10:
                history.pop(0)
            
            # Get most common gesture from history
            text = max(set(history), key=history.count)
        elif len(result.multi_hand_landmarks) == 1:
            text = "Show Both Hands"
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    
    # Draw UI
    cv2.rectangle(frame, (0, 0), (400, 100), (0, 0, 0), -1)
    
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.putText(frame, f"Gesture: {text}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    cv2.imshow("ISL Recognition (2 Hands)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()