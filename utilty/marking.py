import os
import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def create_folders(holders: np.array, videos: int, base_dir='collection'):
    """
    :param base_dir: base folder that will hold everything
    :param holders: ndarray, a list of root folders name to be created
    :param videos: int, how many videos to take and create sub folders in range of int
    :return: make sure to create folders that are needed. and prints if everything is successful.
    """
    base_dir = base_dir
    for holder in holders:
        holder_path = os.path.join(base_dir, holder)
        if not os.path.exists(holder_path):
            os.makedirs(holder_path)
        for i in range(videos):
            suborder_path = os.path.join(holder_path, str(i))
            if not os.path.exists(suborder_path):
                os.makedirs(suborder_path)
        print(f"Created folder '{holder_path}' with sub folders '0' to '29'.")


def get_positions(results):
    """
    gets landmarks positions and saves in concatenated np array
    :param results:
    :return:
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


def save_as_nparray(array: np.array, path):
    """
    :param array: np.array, array to be saved as 'npy' file
    :param path:
    :return:
    """
    save = get_positions(results=array)
    np.save(path, save)


def landmark_detect(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
    return image, results


class Marking:
    main_folder = 'collection'

    def __init__(self):
        self.mp_holistic = mp_holistic
        self.mp_drawing = mp_drawing

    def draw_landmarks(self, image, results):
        # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                                       self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                       self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                       )
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                       )
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                       )
        # Draw right hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                       )
