import gradio as gr
from utilty import marking
import chunk
import cv2
import keras
import numpy as np

# Load model and necessary data
folder_names = np.load('../model/folder_names.npy')
model = keras.models.load_model('../model/TSL_model.keras')


# Helper function to process a single frame
def process_frame(frame):
    with marking.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image, results = marking.landmark_detect(frame, holistic)
        chunk.landmark_draw(image, results, marking.mp_drawing, marking.mp_holistic)
        points = marking.get_positions(results)
    return points


# Function to process video and return the output
def process_video(video_file):
    sequences = []
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        points = process_frame(frame)
        sequences.append(points)
        sequences = sequences[-30:]

        if len(sequences) == 30:
            res = model.predict(np.expand_dims(sequences, axis=0))[0]
            prediction = folder_names[np.argmax(res)]
            break

    cap.release()
    return prediction


# Gradio interface
def predict(video):
    result = process_video(video)
    return result


iface = gr.Interface(
    fn=predict,
    inputs=gr.Video(label="Upload a video"),
    outputs=gr.Textbox(label="Prediction"),
    title="Sign Language Prediction",
    description="Upload a video to get the prediction of the sign language."
)

if __name__ == '__main__':
    iface.launch()
