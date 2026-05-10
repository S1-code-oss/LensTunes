import cv2
import numpy as np
import pygame
import os
import random
from tensorflow.keras.models import load_model
import threading
import time


model = load_model('emotion_model.keras')
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


mood_to_folder = {
    'happy':    'music/happy/',
    'sad':      'music/sad/',
    'angry':    'music/angry/',
    'neutral':  'music/neutral/',
    'fear':     'music/fear/',
    'surprise': 'music/surprise/',
    'disgust':  'music/disgust/'
}

for folder in mood_to_folder.values():
    os.makedirs(folder, exist_ok=True)

pygame.mixer.init()
FADE_STEPS = 20
FADE_DELAY = 0.05
MAX_VOLUME = 1.0

def fade_out():
    volume = pygame.mixer.music.get_volume()
    step = volume / FADE_STEPS
    for _ in range(FADE_STEPS):
        volume = max(0.0, volume - step)
        pygame.mixer.music.set_volume(volume)
        time.sleep(FADE_DELAY)
    pygame.mixer.music.stop()

def fade_in(folder):
    songs = [s for s in os.listdir(folder) if s.endswith('.mp3')]
    if not songs:
        return "no song found"
    song = random.choice(songs)
    pygame.mixer.music.load(folder + song)
    pygame.mixer.music.set_volume(0.0)
    pygame.mixer.music.play()
    step = MAX_VOLUME / FADE_STEPS
    volume = 0.0
    for _ in range(FADE_STEPS):
        volume = min(MAX_VOLUME, volume + step)
        pygame.mixer.music.set_volume(volume)
        time.sleep(FADE_DELAY)
    return song

def transition_music(new_mood):
    fade_out()
    song = fade_in(mood_to_folder[new_mood])
    print(f"Mood: {new_mood} → Playing: {song}")


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

current_emotion = ""
is_transitioning = False
frame_count = 0

print("App running! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

      
        if frame_count % 15 == 0:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            face_img = np.expand_dims(face_img, axis=-1)

            pred = model.predict(face_img, verbose=0)
            new_emotion = emotions[np.argmax(pred)]

            if new_emotion != current_emotion and not is_transitioning:
                current_emotion = new_emotion

        

                def run_transition(mood):
                    global is_transitioning
                    is_transitioning = True
                    transition_music(mood)
                    is_transitioning = False

                t = threading.Thread(target=run_transition, args=(current_emotion,), daemon=True)
                t.start()

        cv2.putText(frame, f'Mood: {current_emotion}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
 

    cv2.imshow('LensTunes', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("App closed.")
