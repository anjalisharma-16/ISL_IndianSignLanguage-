import cv2
import mediapipe as mp
import csv
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

file_name = "isl_data.csv"
current_label = "I"   # 🔁 CHANGE THIS for each gesture

max_samples = 200
count = 0

cap = cv2.VideoCapture(0)

print(f"Collecting {max_samples} samples for: {current_label}")

with open(file_name, mode='a', newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        landmarks = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

            # Save only if both hands detected
            if len(landmarks) == 126 and count < max_samples:
                writer.writerow([current_label] + landmarks)
                count += 1

                print(f"Saved {count}/{max_samples}")
                time.sleep(0.3)

        # Display progress
        cv2.putText(frame, f"{current_label}: {count}/{max_samples}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Collect Data", frame)

        if count >= max_samples:
            print("✅ Done collecting!")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()