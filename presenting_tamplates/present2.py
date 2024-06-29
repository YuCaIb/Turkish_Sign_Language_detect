import gradio as gr
import cv2
from utilty import marking
import chunk
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
    return image, points


# Function to process video frames in real-time
sequences = []


def process_live_frame():
    global sequences
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image, points = process_frame(frame)
        sequences.append(points)
        sequences = sequences[-30:]

        if len(sequences) == 30:
            res = model.predict(np.expand_dims(sequences, axis=0))[0]
            prediction = folder_names[np.argmax(res)]
            cv2.putText(image, prediction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert image from BGR to RGB for Gradio display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the image with the predicted label
        cv2.imshow('Sign Language Prediction', image_rgb)

        # Wait for a key press and handle key events
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "Real-Time Sign Language Prediction Stopped"


# Gradio interface setup
iface = gr.Interface(
    fn=process_live_frame,
    inputs=[],
    outputs=gr.Textbox(label="Prediction", placeholder="Predicted label will appear here"),
    title="Real-Time Sign Language Prediction",
    description="Use your webcam to get real-time predictions of sign language."
)

if __name__ == '__main__':
    iface.launch()
