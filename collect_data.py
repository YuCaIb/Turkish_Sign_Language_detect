import os.path

import utilty
import numpy as np
import cv2

if __name__ == "__main__":
    holders = np.array(['anne', 'bebek', 'misafir', 'sevgili', 'dikkat'])

    videos = 30
    utilty.create_folders(holders, videos)

    # videos will be 30 sequences
    sequence_length = 30
    cap = cv2.VideoCapture(0)

    with utilty.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistics:
        # loop through all root folders
        for holds in holders:
            # loop through all sub folders
            for sequence in range(videos):

                for frame_num in range(sequence_length):
                    ret, frame = cap.read()

                    image, results = utilty.landmark_detect(frame, holistics)
                    print(results)

                    utilty.draw_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, f'start collect data', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'collecting, frames: {holds} , video number {videos}', (60, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 4, cv2.LINE_AA)
                        cv2.waitKey(3000)
                    else:
                        cv2.putText(image, f'collecting, frames: {holds} , video number {videos}', (60, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 4, cv2.LINE_AA)

                    print("frame num", frame_num)
                    print("sequence", sequence)
                    path = os.path.join(utilty.main_folder, holds, str(sequence),
                                        str(frame_num))
                    positions = utilty.get_positions(results)
                    np.save(path, positions)


                    cv2.imshow('feed', image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break


        cap.release()
        cv2.destroyAllWindows()
