
import cv2
import mediapipe as mp
import time

model_path = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

with HandLandmarker.create_from_options(options) as landmarker:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Webcam not detected!")
        raise SystemExit

    start_time = time.monotonic()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    # Draw the 21 hand landmarks
                    for point in hand:
                        x = int(point.x * width)
                        y = int(point.y * height)

                        cv2.circle(
                            frame, (x, y), 6,
                            (0, 255, 0), -1
                        )

                    cv2.putText(
                        frame, "HAND DETECTED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2
                    )
            else:
                cv2.putText(
                    frame, "SHOW YOUR HAND",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2
                )

            cv2.imshow(
                "GestureTune AI - Hand Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
