import cv2
import numpy as np
import pygame
import os
import random
from tensorflow.keras.models import load_model


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

def play_music_for_mood(mood):
    folder = mood_to_folder[mood]
    songs = [s for s in os.listdir(folder) if s.endswith('.mp3')]
    if songs:
        song = random.choice(songs)
        pygame.mixer.music.load(folder + song)
        pygame.mixer.music.play()
        return song
    return "no song found"


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

current_emotion = ""
current_song = ""
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

            if new_emotion != current_emotion:
                current_emotion = new_emotion
                current_song = play_music_for_mood(current_emotion)
                print(f"Mood: {current_emotion} → Playing: {current_song}")

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
