import cv2
import numpy as np
from config import MODEL_INPUT_SIZE, FRAME_SAMPLE_RATE, SEQUENCE_LENGTH

def extract_frame_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_SAMPLE_RATE == 0:
            frame = cv2.resize(frame, MODEL_INPUT_SIZE)
            frame = frame / 255.0
            frames.append(frame)

        frame_id += 1

    cap.release()

    frames = np.array(frames)

    if len(frames) < SEQUENCE_LENGTH:
        padding = np.zeros((SEQUENCE_LENGTH - len(frames), 224, 224, 3))
        frames = np.vstack([frames, padding])
    else:
        frames = frames[:SEQUENCE_LENGTH]

    return np.expand_dims(frames, axis=0)