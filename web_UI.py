from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from utilty import marking
import chunk
import keras

app = Flask(__name__)

# Load your model and other necessary data
folder_names = np.load('model/folder_names.npy')
model = keras.models.load_model('model/TSL_model.keras')
sequences, sentence, predictions, threshold = [], [], [], 0.5  # Initialize sequences here


# Video streaming function
def gen_frames():
    cap = cv2.VideoCapture(0)
    sequences = []  # Move initialization inside the function
    with marking.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = marking.landmark_detect(frame, holistic)
            chunk.landmark_draw(image, results, marking.mp_drawing, marking.mp_holistic)

            points = marking.get_positions(results)
            sequences.append(points)
            sequences = sequences[-30:]

            if len(sequences) == 30:
                res = model.predict(np.expand_dims(sequences, axis=0))[0]
                prediction = folder_names[np.argmax(res)]
                predictions.append(prediction)
                print(prediction)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_predictions')
def get_predictions():
    global predictions
    return jsonify({'predictions': predictions})


if __name__ == '__main__':
    app.run(debug=True)
