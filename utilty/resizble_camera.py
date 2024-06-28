# utilty/resizble_camera.py
import cv2


def resize_and_display(frame, window_name):
    """
    provides resizing camera dynamically.
    :param frame: frame
    :param window_name: string
                        define window's name
    :return:
    """
    height, width = frame.shape[:2]
    # Get the current window size
    new_width = cv2.getWindowImageRect(window_name)[2]
    new_height = cv2.getWindowImageRect(window_name)[3]

    # Calculate the new aspect ratio
    aspect_ratio = width / height
    new_aspect_ratio = new_width / new_height

    # Adjust the frame size to maintain the aspect ratio
    if new_aspect_ratio > aspect_ratio:
        new_width = int(new_height * aspect_ratio)
    else:
        new_height = int(new_width / aspect_ratio)

    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    cv2.imshow(window_name, resized_frame)


class ResizableCamera:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

    def start_camera(self, window_name='Resizable Camera Window'):
        """
        Start the camera and display the resized frame.
        :param window_name: string
                            define window's name
        :return:
        """
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width, self.height)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resize_and_display(frame, window_name)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Test the class
if __name__ == "__main__":
    camera = ResizableCamera()
    camera.start_camera()
