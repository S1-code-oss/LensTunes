import cv2
import numpy as np
import pygame
import os
import random
import threading
import time
from flask import Flask, Response, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ── Model ──
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

# ── Pygame ──
pygame.mixer.init()
FADE_STEPS = 20
FADE_DELAY = 0.05
MAX_VOLUME = 1.0

# ── Shared state ──
state = {
    'emotion': 'neutral',
    'song': '',
    'is_transitioning': False,
    'face_detected': False,
    'confidence': 0.0,
}

# ── Music ──
def fade_out():
    volume = pygame.mixer.music.get_volume()
    step = volume / FADE_STEPS if volume > 0 else 0
    for _ in range(FADE_STEPS):
        volume = max(0.0, volume - step)
        pygame.mixer.music.set_volume(volume)
        time.sleep(FADE_DELAY)
    pygame.mixer.music.stop()

def fade_in(folder):
    songs = [s for s in os.listdir(folder) if s.endswith('.mp3')]
    if not songs:
        return 'no song found'
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
    state['song'] = song
    print(f"Mood: {new_mood} -> Playing: {song}")

# ── Camera thread ──
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
current_emotion = ''
frame_count = 0
latest_frame = None
frame_lock = threading.Lock()

def camera_loop():
    global current_emotion, frame_count, latest_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        state['face_detected'] = len(faces) > 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (192, 122, 212), 2)

            if frame_count % 15 == 0:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (48, 48))
                face_img = face_img / 255.0
                face_img = np.expand_dims(face_img, axis=0)
                face_img = np.expand_dims(face_img, axis=-1)

                pred = model.predict(face_img, verbose=0)
                new_emotion = emotions[np.argmax(pred)]
                state['confidence'] = float(np.max(pred))

                if new_emotion != current_emotion and not state['is_transitioning']:
                    current_emotion = new_emotion
                    state['emotion'] = current_emotion

                    def run_transition(mood):
                        state['is_transitioning'] = True
                        transition_music(mood)
                        state['is_transitioning'] = False

                    t = threading.Thread(target=run_transition, args=(current_emotion,), daemon=True)
                    t.start()

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            latest_frame = buffer.tobytes()

        frame_count += 1
        time.sleep(0.03)

threading.Thread(target=camera_loop, daemon=True).start()

# ── Routes ──
def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    return jsonify(state)

if __name__ == '__main__':
    print("LensTunes running at http://localhost:5000")
    app.run(debug=False, threaded=True, port=5000)
