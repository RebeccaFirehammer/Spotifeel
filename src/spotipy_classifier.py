from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

import spotipy
import spotipy.util as util
import time
import requests
import json

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

########
username = '1291327497'
playlist_id = '6xJ5BRhgKyN3nHTxyTd9Xl'
scope = 'playlist-modify-public playlist-read-private playlist-modify-private user-read-currently-playing streaming'
client_id = '0ff658e7b53546378a4088919b0c22b9'
client_secret = 'fd5c35fbddf645a0aeeb2289f4c15e59'
redirect_uri = 'https://spotify.com'
########

token = util.prompt_for_user_token(username,
                                   scope,
                                   client_id=client_id,
                                   client_secret=client_secret,
                                   redirect_uri=redirect_uri)
sp = spotipy.Spotify(auth=token)

headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + token,
}

params = (('market', 'US'),)


def play_next_track():
    response = requests.post('https://api.spotify.com/v1/me/player/next', headers=headers, params=params)
    return


def get_current_song():
    response = requests.get('https://api.spotify.com/v1/me/player/currently-playing', headers=headers, params=params)
    data = json.loads(response.text)
    current_song = data["item"]["id"]
    return current_song


def is_on_playlist(current_song, playlist_name):
    playlists = sp.user_playlists(username)
    for playlist in playlists['items']:
        if playlist['name'] == playlist_name:
            results = sp.user_playlist(username, playlist['id'], fields="tracks,next")
            tracks = results['tracks']
            for i, item in enumerate(tracks['items']):
                track = item['track']
                if current_song == track['id']:
                    return True
            return False

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
delay = 10
angry_delay = time.time()
happy_delay = time.time()
surprised_delay = time.time()
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
            if angry_delay > time.time():
                continue
            play_next_track()
            angry_delay = time.time() + delay
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
            if happy_delay > time.time():
                continue
            current_song = get_current_song()
            if not is_on_playlist(current_song, "Spotifeel"):
                sp.user_playlist_add_tracks(username, playlist_id, [current_song])
            happy_delay = time.time() + delay
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
            if surprised_delay > time.time():
                continue
            current_song = get_current_song()
            if not is_on_playlist(current_song, "Spotifeel"):
                sp.user_playlist_add_tracks(username, playlist_id, [current_song])
            surprised_delay = time.time() + delay
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
