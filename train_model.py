import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

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

df = pd.read_csv("isl_data.csv", header=None)

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

X = np.array([normalize_landmarks(row) for row in X])

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2)

model = Sequential([
    Dense(256, activation='relu', input_shape=(126,)),  # 2 hands = 126
    Dense(128, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)

loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

model.save("model.h5")
np.save("labels.npy", le.classes_)
import matplotlib.pyplot as plt

# Accuracy graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()

# Loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()