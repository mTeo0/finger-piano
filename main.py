# main modules
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# piano modules
import pygame
import threading
import json

# config
MODEL_PATH = "hand_landmarker.task"

pygame.mixer.init()

options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

estado_anterior = {
    "pulgar": None,
    "indice": None,
    "medio": None,
    "anular": None,
    "meñique": None
}

with open('asigned_notes.json') as file:
    asigned_notes = json.load(file)

def play_sound(note_file):
    def _play():
        sound = pygame.mixer.Sound(f"notes/{note_file}.wav")
        sound.play()
    threading.Thread(target=_play, daemon=True).start()

def dedo_vertical(hand, tip, pip):
    return hand[tip].y < hand[pip].y

def pulgar(hand):
    return hand[4].x > hand[3].x

def puño(estados):
    return all(not v for v in estados.values())

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, frame_id)
    frame_id += 1

    texto = []

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        estados = {
            "pulgar": pulgar(hand),
            "indice": dedo_vertical(hand, 8, 6),
            "medio": dedo_vertical(hand, 12, 10),
            "anular": dedo_vertical(hand, 16, 14),
            "meñique": dedo_vertical(hand, 20, 18)
        }



        for dedo, estado in estados.items():
            prev = estado_anterior[dedo]

            if prev is not None:
                if not prev and estado:
                    print(f"⬆️ Subiste {dedo}")
                elif prev and not estado:
                    print(f"⬇️ Bajaste {dedo}")
                    play_sound(asigned_notes[dedo])

            estado_anterior[dedo] = estado
            texto.append(f"{dedo}: {'Arriba' if estado else 'Abajo'}")

        if puño(estados):
            texto.append("✊ PUÑO CERRADO")
            play_sound(asigned_notes['puño'])

    y = 30
    for t in texto:
        cv2.putText(frame, t, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30

    cv2.imshow("MediaPipe Tasks - Dedos", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
