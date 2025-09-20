import cv2
import imutils
import numpy as np
import argparse


# Initialize HOG descriptor/person detector
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect(frame):
    """Detect humans in the given frame"""
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(
        frame, winStride=(4, 4), padding=(8, 8), scale=1.03
    )

    person = 1
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f'Person {person}',
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        person += 1

    cv2.putText(
        frame, 'Status : Detecting', (40, 40),
        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2
    )
    cv2.putText(
        frame, f'Total Persons : {person - 1}', (40, 70),
        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2
    )
    cv2.imshow('output', frame)

    return frame


def humanDetector(args):
    """Main function to choose between image, video or camera"""
    image_path = args["image"]
    video_path = args["video"]
    camera = args["camera"]

    writer = None

    if args['output'] is not None and image_path is None:
        # Writer will be initialized later after knowing frame size
        writer = args['output']

    if camera:
        print('[INFO] Opening Web Cam...')
        detectByCamera(writer)
    elif video_path is not None:
        print('[INFO] Opening Video...')
        detectByPathVideo(video_path, writer)
    elif image_path is not None:
        print('[INFO] Opening Image...')
        detectByPathImage(image_path, args['output'])
    else:
        print("[ERROR] No input source provided. Use --camera or --video or --image")


def detectByCamera(output_path=None):
    """Detect humans from webcam"""
    video = cv2.VideoCapture(0)
    print('Detecting people...')

    writer = None

    while True:
        check, frame = video.read()
        if not check:
            break

        frame = detect(frame)

        # Initialize writer with frame size if output_path is given
        if output_path is not None and writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (width, height)
            )

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def detectByPathVideo(path, output_path=None):
    """Detect humans from a video file"""
    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if not check:
        print('Video Not Found. Please Enter a Valid Path.')
        return

    print('Detecting people...')
    writer = None

    while video.isOpened():
        check, frame = video.read()
        if not check:
            break

        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        frame = detect(frame)

        if output_path is not None and writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (width, height)
            )

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def detectByPathImage(path, output_path=None):
    """Detect humans from an image"""
    image = cv2.imread(path)
    if image is None:
        print("Image not found. Please check the path.")
        return

    image = imutils.resize(image, width=min(800, image.shape[1]))
    result_image = detect(image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def argsParser():
    """Parse command line arguments"""
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File")
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File")
    arg_parse.add_argument(
        "-c", "--camera", action="store_true", help="Use camera for real-time detection"
    )
    arg_parse.add_argument(
        "-o", "--output", type=str, default=None, help="path to save output (image/video)"
    )
    args = vars(arg_parse.parse_args())
    return args


if __name__ == "__main__":
    args = argsParser()
    humanDetector(args)
