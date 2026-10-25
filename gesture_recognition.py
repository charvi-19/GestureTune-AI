
import cv2
import mediapipe as mp
import pygame
import time
import glob
import os

# ==========================================
# 1. MUSIC PLAYER SETUP
# ==========================================

MUSIC_FOLDER = "music"

songs = sorted(
    glob.glob(os.path.join(MUSIC_FOLDER, "*.mp3"))
)

if not songs:
    raise FileNotFoundError(
        "Add at least one MP3 file to the music folder."
    )

pygame.mixer.init()

current_song_index = 0
is_paused = False
is_muted = False
volume = 0.5

pygame.mixer.music.set_volume(volume)


def play_music():
    global is_paused

    if is_paused:
        pygame.mixer.music.unpause()
        is_paused = False
        print("Music resumed!")

    elif not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(songs[current_song_index])
        pygame.mixer.music.play()
        is_paused = False
        print("Playing:", os.path.basename(
            songs[current_song_index]
        ))

    else:
        print("Music is already playing.")


def pause_music():
    global is_paused

    if is_paused:
        print("Music is already paused.")

    elif pygame.mixer.music.get_busy():
        pygame.mixer.music.pause()
        is_paused = True
        print("Music paused.")

    else:
        print("No music is currently playing.")


def next_song():
    global current_song_index, is_paused

    current_song_index = (
        current_song_index + 1
    ) % len(songs)

    pygame.mixer.music.load(songs[current_song_index])
    pygame.mixer.music.play()
    is_paused = False

    print("Next:", os.path.basename(
        songs[current_song_index]
    ))


def previous_song():
    global current_song_index, is_paused

    current_song_index = (
        current_song_index - 1
    ) % len(songs)

    pygame.mixer.music.load(songs[current_song_index])
    pygame.mixer.music.play()
    is_paused = False

    print("Previous:", os.path.basename(
        songs[current_song_index]
    ))


def toggle_mute():
    global is_muted

    is_muted = not is_muted

    if is_muted:
        pygame.mixer.music.set_volume(0)
        print("Muted.")
    else:
        pygame.mixer.music.set_volume(volume)
        print("Unmuted. Volume:", int(volume * 100), "%")


def change_volume(new_volume):
    global volume

    volume = max(0.0, min(1.0, new_volume))

    if not is_muted:
        pygame.mixer.music.set_volume(volume)


# ==========================================
# 2. MEDIAPIPE SETUP - TWO HANDS
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)


# ==========================================
# 3. FINGER AND GESTURE RECOGNITION
# ==========================================

def finger_states(landmarks):
    """
    Return whether index, middle, ring and pinky
    fingers are raised.

    Best suited to a palm facing the camera.
    """

    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y

    return [
        index_up,
        middle_up,
        ring_up,
        pinky_up
    ]


def recognize_right_hand(landmarks):
    fingers = finger_states(landmarks)
    count = sum(fingers)

    index_up, middle_up, ring_up, pinky_up = fingers

    if count == 0:
        return "FIST"

    if count == 1 and index_up:
        return "PLAY"

    if (
        count == 2
        and index_up
        and middle_up
    ):
        return "NEXT"

    if count == 3:
        return "PREVIOUS"

    if count == 4:
        return "MUTE"

    return "UNKNOWN"


def handle_right_gesture(gesture):
    if gesture == "PLAY":
        play_music()

    elif gesture == "FIST":
        pause_music()

    elif gesture == "NEXT":
        next_song()

    elif gesture == "PREVIOUS":
        previous_song()

    elif gesture == "MUTE":
        toggle_mute()


# ==========================================
# 4. VOLUME CONTROL
# ==========================================

def calculate_volume(landmarks):
    """
    The left hand controls volume.

    Raise the index finger while keeping the
    middle, ring and pinky fingers folded.
    Move thumb and index fingertips apart to
    increase volume; bring them together to
    decrease it.
    """

    fingers = finger_states(landmarks)

    index_up, middle_up, ring_up, pinky_up = fingers

    volume_gesture = (
        index_up
        and not middle_up
        and not ring_up
        and not pinky_up
    )

    if not volume_gesture:
        return None

    thumb = landmarks[4]
    index = landmarks[8]

    distance = (
        (thumb.x - index.x) ** 2
        + (thumb.y - index.y) ** 2
    ) ** 0.5

    # Map fingertip distance to 0-100% volume.
    # Adjust these limits if needed.
    min_distance = 0.025
    max_distance = 0.25

    normalized = (
        (distance - min_distance)
        / (max_distance - min_distance)
    )

    return max(0.0, min(1.0, normalized))


# ==========================================
# 5. STABILIZATION SETTINGS
# ==========================================

STABLE_FRAMES = 8
COOLDOWN_SECONDS = 1.0

candidate_gesture = None
candidate_count = 0
last_triggered_gesture = None
last_action_time = 0

status_text = "Show your hands"


# ==========================================
# 6. START THE WEBCAM
# ==========================================

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError("Webcam not detected!")

        start_time = time.monotonic()

        print("\nGestureTune AI started!")
        print("RIGHT HAND:")
        print("  One finger  = Play / Resume")
        print("  Fist        = Pause")
        print("  Two fingers = Next song")
        print("  Three       = Previous song")
        print("  Open palm   = Mute / Unmute")
        print("\nLEFT HAND:")
        print("  Index raised + thumb movement = Volume")
        print("\nKeyboard shortcuts:")
        print("  Space = Play / Pause")
        print("  Right arrow = Next song")
        print("  Left arrow = Previous song")
        print("  M = Mute / Unmute")
        print("  Q = Quit\n")

        try:
            while True:
                success, frame = camera.read()

                if not success:
                    print("Unable to read webcam.")
                    break

                frame = cv2.flip(frame, 1)

                height, width, _ = frame.shape

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = int(
                    (time.monotonic() - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                right_gesture = None
                volume_value = None

                # ----------------------------------
                # 7. PROCESS BOTH HANDS
                # ----------------------------------

                if result.hand_landmarks:

                    for i, landmarks in enumerate(
                        result.hand_landmarks
                    ):
                        hand_label = "Unknown"

                        # Read MediaPipe's handedness result
                        if (
                            result.handedness
                            and i < len(result.handedness)
                            and result.handedness[i]
                        ):
                            hand_label = (
                                result.handedness[i][0].category_name
                            )

                        # Draw landmarks
                        for point in landmarks:
                            x = int(point.x * width)
                            y = int(point.y * height)

                            cv2.circle(
                                frame,
                                (x, y),
                                4,
                                (0, 255, 0),
                                -1
                            )

                        # Draw connections
                        for connection in (
                            mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                        ):
                            p1_landmark = landmarks[connection.start]
                            p2_landmark = landmarks[connection.end]

                            p1 = (
                                int(p1_landmark.x * width),
                                int(p1_landmark.y * height)
                            )

                            p2 = (
                                int(p2_landmark.x * width),
                                int(p2_landmark.y * height)
                            )

                            cv2.line(
                                frame,
                                p1,
                                p2,
                                (255, 150, 0),
                                2
                            )

                        # LEFT HAND = VOLUME
                        if hand_label == "Left":

                            new_volume = calculate_volume(
                                landmarks
                            )

                            if new_volume is not None:
                                volume_value = new_volume
                                change_volume(new_volume)

                                cv2.putText(
                                    frame,
                                    "LEFT: VOLUME CONTROL",
                                    (15, 145),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 0),
                                    2
                                )

                        # RIGHT HAND = MUSIC COMMANDS
                        elif hand_label == "Right":

                            right_gesture = recognize_right_hand(
                                landmarks
                            )

                # ----------------------------------
                # 8. STABILIZE RIGHT-HAND GESTURES
                # ----------------------------------

                if right_gesture is not None:

                    if right_gesture == candidate_gesture:
                        candidate_count += 1
                    else:
                        candidate_gesture = right_gesture
                        candidate_count = 1

                    if candidate_count >= STABLE_FRAMES:

                        current_time = time.monotonic()

                        if right_gesture != "UNKNOWN":

                            if (
                                right_gesture
                                != last_triggered_gesture
                            ):
                                if (
                                    current_time - last_action_time
                                    >= COOLDOWN_SECONDS
                                ):
                                    handle_right_gesture(
                                        right_gesture
                                    )

                                    last_triggered_gesture = (
                                        right_gesture
                                    )
                                    last_action_time = current_time

                                    status_text = right_gesture

                else:
                    candidate_gesture = None
                    candidate_count = 0
                    last_triggered_gesture = None

                # ----------------------------------
                # 9. DISPLAY MUSIC INFORMATION
                # ----------------------------------

                song_name = os.path.basename(
                    songs[current_song_index]
                )

                volume_percent = int(volume * 100)

                if is_muted:
                    volume_display = "MUTED"
                else:
                    volume_display = str(volume_percent) + "%"

                cv2.putText(
                    frame,
                    "GestureTune AI",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Action: " + status_text,
                    (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Song: " + song_name[:35],
                    (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Volume: " + volume_display,
                    (15, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 255),
                    2
                )

                # Draw volume bar
                bar_x = 15
                bar_y = 195
                bar_width = 200
                bar_height = 16

                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + bar_width, bar_y + bar_height),
                    (200, 200, 200),
                    2
                )

                filled_width = int(
                    bar_width * volume
                )

                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + filled_width, bar_y + bar_height),
                    (0, 200, 0),
                    -1
                )

                cv2.imshow(
                    "GestureTune AI - Two Hand Music Player",
                    frame
                )

                # ----------------------------------
                # 10. KEYBOARD BACKUP CONTROLS
                # ----------------------------------

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

                elif key == ord(" "):
                    if is_paused or not pygame.mixer.music.get_busy():
                        play_music()
                    else:
                        pause_music()

                elif key == 83 or key == ord("n"):
                    next_song()

                elif key == 81 or key == ord("b"):
                    previous_song()

                elif key == ord("m"):
                    toggle_mute()

                elif key == ord("+") or key == ord("="):
                    change_volume(volume + 0.05)

                elif key == ord("-") or key == ord("_"):
                    change_volume(volume - 0.05)

        finally:
            camera.release()
            cv2.destroyAllWindows()

finally:
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    print("GestureTune AI closed.")
