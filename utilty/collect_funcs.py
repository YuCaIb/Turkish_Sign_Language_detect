import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from utilty import Marking, ResizableCamera
from utilty import marking, resizble_camera


video = 30
seq_len = 30


class DataCollector:
    def __init__(self, videos=video, sequence_length=seq_len):
        self.videos = videos
        self.sequence_length = sequence_length
        self.stop_processing = False
        self.marker = Marking()
        self.camera = ResizableCamera()

    def process_frame(self, frame, holds, sequence, frame_num, holistics):
        if self.stop_processing:
            return None

        image, results = marking.landmark_detect(frame, holistics)
        self.marker.draw_landmarks(image, results)

        if frame_num == 0:
            cv2.putText(image, 'start collect data', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)

        cv2.putText(image, f'collecting, frames: {holds} , video number {sequence}',
                    (60, 120 if frame_num == 0 else 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if frame_num == 0 else (255, 0, 0), 4, cv2.LINE_AA)

        path = os.path.join(self.marker.main_folder, holds, str(sequence), str(frame_num))
        positions = marking.get_positions(results)
        np.save(path, positions)

        return image

    def collect_data(self, holistics, cap, window_name, holders):
        self.stop_processing = False

        with ThreadPoolExecutor() as executor:
            for holds in holders:
                for sequence in range(self.videos):
                    futures = []
                    for frame_num in range(self.sequence_length):
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        future = executor.submit(self.process_frame, frame, holds, sequence, frame_num, holistics)
                        futures.append(future)

                        # Display the frame
                        image = future.result()
                        if image is not None:
                            resizble_camera.resize_and_display(image, window_name)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.stop_processing = True
                            break

                    # Wait for all frames in the current video to be processed
                    for future in futures:
                        future.result()

                    if self.stop_processing:
                        break

                    # Wait for 'u' key press to proceed to the next video
                    print("Press 'u' to proceed to the next video...")
                    while True:
                        if cv2.waitKey(1) & 0xFF == ord('u'):
                            break

                if self.stop_processing:
                    break
