import cv2
import numpy as np
import pygame
import os
import random
from tensorflow.keras.models import load_model
import io
from collections import deque
from pydub import AudioSegment
from pydub.effects import speedup


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


pygame.mixer.init()

def apply_intensity_effects(filepath, emotion, intensity):
    audio = AudioSegment.from_mp3(filepath)

    if emotion in ['angry', 'fear', 'surprise']:
        speed_factor = 1.0 + (intensity - 0.5) * 0.8
    elif emotion in ['sad', 'disgust']:
        speed_factor = 1.0 - (intensity - 0.5) * 0.6
    else:
        speed_factor = 1.0

    if abs(speed_factor - 1.0) > 0.01:
        audio = speedup(audio, playback_speed=max(speed_factor, 0.5))

    if emotion == 'angry':
        db_change = (intensity - 0.5) * 10
    elif emotion == 'sad':
        db_change = -(intensity * 6)
    else:
        db_change = 0
    audio = audio + db_change

    if emotion in ['angry', 'fear'] and intensity > 0.6:
        bass = audio.low_pass_filter(200)
        audio = audio.overlay(bass - 8)

    buf = io.BytesIO()
    audio.export(buf, format='mp3')
    buf.seek(0)
    pygame.mixer.music.load(buf)
    pygame.mixer.music.play()


def pick_and_play(mood, intensity):
    folder = mood_to_folder[mood]
    if not os.path.isdir(folder):
        return None
    songs = [s for s in os.listdir(folder) if s.endswith('.mp3')]
    if not songs:
        pygame.mixer.music.stop()
        return None
    song = random.choice(songs)
    try:
        apply_intensity_effects(folder + song, mood, intensity)
        return song
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

current_emotion = ""
current_song = ""
frame_count = 0

last_intensity      = 0.0
INTENSITY_THRESHOLD = 0.15
emotion_history     = deque(maxlen=10)
intensity_history   = deque(maxlen=10)

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
            pred          = model.predict(face_img, verbose=0)[0]
            idx           = int(np.argmax(pred))
            raw_emotion   = emotions[idx]
            raw_intensity = float(pred[idx])

            emotion_history.append(raw_emotion)
            intensity_history.append(raw_intensity)

             new_emotion        = max(set(emotion_history), key=emotion_history.count)
             smoothed_intensity = sum(intensity_history) / len(intensity_history)

if new_emotion != current_emotion or abs(smoothed_intensity - last_intensity) > INTENSITY_THRESHOLD:
    current_emotion = new_emotion
    last_intensity  = smoothed_intensity
    current_song    = pick_and_play(current_emotion, smoothed_intensity) or "no song found"
    print(f"Mood: {current_emotion} | Intensity: {smoothed_intensity:.0%} → {current_song}")

       label = f'{current_emotion} ({last_intensity:.0%})'
            cv2.putText(frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f'Song: {current_song}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow('LensTunes', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("App closed.")
