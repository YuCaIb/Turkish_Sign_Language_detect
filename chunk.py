## we don't use it I just left it there so people may like to see way I trying things.


# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils
import os
import cv2
import mediapipe as mp
import numpy as np


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


# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print('cap is not available')
#     exit()

# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.3, model_complexity=2) as holistic:
#     while True:
#         ret, frame = cap.read()
#
#         if not ret:
#             print("can't receive frame")
#             break
#         # operations on the frame need to filled there
#         image, results = landmark_detect(frame, holistic)
#         print(results)
#         # Display
#         landmark_draw(image, results)
#         cv2.imshow('feed', image)
#
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
# len(results.left_hand_landmarks.landmark)  # 21 mark
# len(results.right_hand_landmarks.landmark)  # 21 mark
# len(results.face_landmarks.landmark)  # 468 mark
# len(results.pose_landmarks.landmark)  # 33 mark
#
# print(results.pose_landmarks)
# print(results.right_hand_landmarks)  # 468 mark
# print(results.left_hand_landmarks)  # 468 mark
#
# pose = []
# for res in results.pose_landmarks.landmark:
#     test = np.array([res.x, res.y, res.z, res.visibility])
#     pose.append(test)
#
# len(pose)
#
#
# def extract_keypoint(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
#                      results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
#     face = np.array([[res.x, res.y, res.z] for res in
#                      results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
#     rh = np.array([[res.x, res.y, res.z] for res in
#                    results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
#         21 * 3)
#     lh = np.array([[res.x, res.y, res.z] for res in
#                    results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
#     return np.concatenate([pose, face, rh, lh])

#
# dir_export_arrays = os.path.join('extracted_data')
# actions_list = np.array(['günaydın', 'merhaba', 'kardeş', 'abi', 'abla'])

# no_sequences = 30
#
# sequence_length = 30
#
# for action in actions_list:
#     for sequence in range(no_sequences):
#         os.makedirs(os.path.join(dir_export_arrays, action, str(sequence)), exist_ok=True)

"""Thread : 
Thread Usage, example, differnces.

import time

def square_numbers(numbers):
    for number in numbers:
        time.sleep(1)  # Simulate a time-consuming task
        print(f'Square of {number} is {number * number}')

numbers = [1, 2, 3, 4, 5]
start_time = time.time()
square_numbers(numbers)
end_time = time.time()

print(f'Time taken without threads: {end_time - start_time:.2f} seconds')

import time
import threading

def square_number(number):
    time.sleep(1)  # Simulate a time-consuming task
    print(f'Square of {number} is {number * number}')

numbers = [1, 2, 3, 4, 5]
threads = []

start_time = time.time()
for number in numbers:
    thread = threading.Thread(target=square_number, args=(number,))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

end_time = time.time()

print(f'Time taken with threads: {end_time - start_time:.2f} seconds')


Conclusion, As a result, using threads can improve performance and speed up processes,
but this benefit comes with a more complex code structure and requires careful management.
Therefore, it is important to be cautious when using threads and to apply proper synchronization techniques.
"""

"""
if __name__ == "__main__":
    print('12')

import pandas

df = pandas.read_csv('done.txt')

"""
