from utilty import marking
import chunk
import cv2
import keras
import numpy as np


folder_names = np.load('../model/folder_names.npy')
model = keras.models.load_model('../model/TSL_model.keras')
sequences, sentence, predictions, threshold = [], [], [], 0.5

cap = cv2.VideoCapture(0)


if __name__ == '__main__':
    with marking.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = marking.landmark_detect(frame, holistic)

            # Draw landmarks
            chunk.landmark_draw(image, results, marking.mp_drawing, marking.mp_holistic)

            # get array
            points = marking.get_positions(results)
            # always get last 30 frame
            sequences.append(points)
            sequences = sequences[-30:]

            if len(sequences) == 30:
                res = model.predict(np.expand_dims(sequences, axis=0))[0]
                print(folder_names[np.argmax(res)])
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
