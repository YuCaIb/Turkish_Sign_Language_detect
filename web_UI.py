from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from utilty import marking
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
            landmark_draw(image, results, marking.mp_drawing, marking.mp_holistic)

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



def landmark_detect(image, model):
    """

    :param image: image
    :param model: insert mediapipe model
    :return: image and results,
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # Color conversion bgr 2 rgb
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # rgb to bgr
    return image, results


def landmark_draw(image, results, mp_drawing, mp_holistic):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


if __name__ == '__main__':
    app.run(debug=True)
