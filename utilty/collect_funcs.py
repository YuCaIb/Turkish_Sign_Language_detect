
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from utilty import marking
from utilty import resizble_camera

videos = 30
sequence_length = 30


# Global flag to control the loop
stop_processing = False


def process_frame(frame, holds, sequence, frame_num, holistics,window_name):
    global stop_processing
    if stop_processing:
        return None

    image, results = marking.landmark_detect(frame, holistics)
    marking.draw_landmarks(image, results)

    if frame_num == 0:
        cv2.putText(image, f'start collect data', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 4, cv2.LINE_AA)

        cv2.putText(image, f'collecting, frames: {holds} , video number {sequence}', (60, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 4, cv2.LINE_AA)

    else:
        cv2.putText(image, f'collecting, frames: {holds} , video number {sequence}', (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 4, cv2.LINE_AA)

    path = os.path.join(marking.main_folder, holds, str(sequence), str(frame_num))
    positions = marking.get_positions(results)
    np.save(path, positions)

    return image


def collect_data(holistics, cap, window_name,holders):
    global stop_processing

    with ThreadPoolExecutor() as executor:
        futures = []
        for holds in holders:
            for sequence in range(videos):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    future = executor.submit(process_frame, frame, holds, sequence, frame_num, holistics, window_name)
                    futures.append(future)

                    # Display the frame
                    image = future.result()
                    if image is not None:
                        resizble_camera.resize_and_display(image, window_name)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_processing = True
                        break

                if stop_processing:
                    break
            if stop_processing:
                break

        for future in futures:
            future.result()  # Ensure all threads have completed
