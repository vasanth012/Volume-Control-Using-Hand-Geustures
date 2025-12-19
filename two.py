import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Function to count raised fingers
def calculate_fingers(hand_landmarks):
    fingers = []
    
    # Thumb (checks horizontal position)
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other four fingers (checks vertical position)
    for id in range(1, 5):
        if hand_landmarks.landmark[mp_hands.HandLandmark(id * 4)].y < hand_landmarks.landmark[mp_hands.HandLandmark(id * 4 - 2)].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# Function to adjust volume
def set_volume(fingers):
    current_volume = volume.GetMasterVolumeLevelScalar()
    num_fingers = sum(fingers)  # Count the number of fingers raised

    if num_fingers == 4:  # Four fingers raised → Volume UP
        new_volume = min(current_volume + 0.05, 1.0)
        volume.SetMasterVolumeLevelScalar(new_volume, None)
        print("🔊 Volume Up Gesture Detected")

    elif num_fingers == 2:  # Two fingers raised → Volume DOWN
        new_volume = max(current_volume - 0.05, 0.0)
        volume.SetMasterVolumeLevelScalar(new_volume, None)
        print("🔉 Volume Down Gesture Detected")

    return volume.GetMasterVolumeLevelScalar()

# Start capturing video
cap = cv2.VideoCapture(0)

# Ensure webcam is accessible
if not cap.isOpened():
    print("⚠️ Error: Could not open webcam!")
    exit()

while True:
    success, img = cap.read()
    
    if not success:
        print("⚠️ Error: Failed to grab frame!")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = calculate_fingers(hand_landmarks)
            current_volume = set_volume(fingers)

            # Draw the volume bar on the left side of the screen
            vol_bar = int(current_volume * 400)  # Scale to fit
            cv2.rectangle(img, (50, 550 - vol_bar), (85, 550), (0, 255, 0), cv2.FILLED)  # Volume fill
            cv2.rectangle(img, (50, 150), (85, 550), (0, 255, 0), 2)  # Volume bar outline
            cv2.putText(img, f'{int(current_volume * 100)}%', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
