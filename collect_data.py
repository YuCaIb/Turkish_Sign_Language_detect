from utilty import marking, collect_funcs
from utilty.resizble_camera import width, height
from utilty.collect_funcs import videos
import numpy as np
import cv2

if __name__ == "__main__":
    holders = np.array(['anne', 'bebek', 'misafir', 'sevgili', 'dikkat'])

    marking.create_folders(holders, videos)

    window_name = 'Data Collector'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    with marking.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistics:
        collect_funcs.collect_data(holistics, cap, window_name, holders)

    cap.release()
    cv2.destroyAllWindows()
