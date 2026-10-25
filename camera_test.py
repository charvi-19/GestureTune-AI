
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Webcam could not be opened.")
    raise SystemExit

print("Webcam started!")
print("Click the camera window and press Q or ESC to exit.")
print("Alternatively, press Ctrl+C in the terminal.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("GestureTune AI - Camera Test", frame)

        key = cv2.waitKey(30)
        key = key & 0xFF

        if key == ord("q") or key == 27:
            print("Closing camera...")
            break

except KeyboardInterrupt:
    print("Camera stopped from terminal.")

finally:
    camera.release()
    cv2.destroyAllWindows()

    # Ensure OpenCV windows are closed
    for _ in range(5):
        cv2.waitKey(1)

    print("Camera released successfully.")
