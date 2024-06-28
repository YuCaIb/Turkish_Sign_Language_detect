import cv2
import numpy as np

from utilty import marking, collect_funcs
from utilty.collect_funcs import video

marker = marking
collector = collect_funcs.DataCollector()

if __name__ == "__main__":
    holders = np.array(["Değerli","Diploma","Dinlemek","Devam", "Denemek"])

    marker.create_folders(holders, video)
    window_name = 'Data Collector'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with marking.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                      model_complexity=1) as holistics:
        collector.collect_data(holistics, cap, window_name, holders)

    cap.release()
    cv2.destroyAllWindows()
