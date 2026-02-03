import cv2
import mediapipe as mp
import numpy as np
import time
import signal
import subprocess
from collections import deque

# Global flag for graceful exit
running = True
ffplay_process = None

def signal_handler(sig, frame):
    global running, ffplay_process
    print("\nInterrupted by user (Ctrl+C). Closing...")
    running = False
    if ffplay_process:
        ffplay_process.terminate()

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def main():
    global running, ffplay_process
    mp_face_mesh = mp.solutions.face_mesh
    
    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    mambo_path = "vid/mambo.mp4"

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            print("Eye tracker started. Press 'q' to exit or Ctrl+C to stop.")
            
            # Timer variables
            looking_down_start_time = None
            TRIGGER_BUFFER = 3.0 # seconds
            DISPLAY_VIDEO = False
            
            # Smoothing variables
            ratio_history = deque(maxlen=5)

            while running:
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                image = cv2.flip(image, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)
                
                # Default values for this frame
                avg_ratio = 0.5
                head_pitch = 0.5
                looking_down_triggered = False
                not_looking_at_pc = False

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        h, w, c = image.shape
                        
                        # --- Gaze Detection ---
                        def get_vertical_ratio(top_idx, bottom_idx, iris_idx, landmarks):
                            top_p = landmarks.landmark[top_idx]
                            bot_p = landmarks.landmark[bottom_idx]
                            iris_p = landmarks.landmark[iris_idx]
                            eye_height = (bot_p.y - top_p.y) * h
                            iris_dist = (iris_p.y - top_p.y) * h
                            return iris_dist / eye_height if eye_height != 0 else 0.5

                        forehead = face_landmarks.landmark[10]
                        nose = face_landmarks.landmark[1]
                        chin = face_landmarks.landmark[152]
                        head_pitch = (nose.y - forehead.y) / (chin.y - forehead.y) if (chin.y - forehead.y) != 0 else 0.5

                        left_ratio = get_vertical_ratio(159, 145, 468, face_landmarks)
                        right_ratio = get_vertical_ratio(386, 374, 473, face_landmarks)
                        
                        current_ratio = (left_ratio + right_ratio) / 2.0
                        ratio_history.append(current_ratio)
                        avg_ratio = sum(ratio_history) / len(ratio_history)
                        
                        # Thresholds
                        IRIS_SENSITIVE_THRESHOLD = 0.54
                        HEAD_PITCH_THRESHOLD = 0.61
                        NOT_LOOKING_THRESHOLD = 0.18
                        
                        looking_down_triggered = (avg_ratio > IRIS_SENSITIVE_THRESHOLD) or (head_pitch > HEAD_PITCH_THRESHOLD)
                        not_looking_at_pc = (avg_ratio < NOT_LOOKING_THRESHOLD)

                # --- Video Trigger Timer ---
                if looking_down_triggered:
                    if looking_down_start_time is None:
                        looking_down_start_time = time.time()
                    elif time.time() - looking_down_start_time > TRIGGER_BUFFER:
                        DISPLAY_VIDEO = True
                else:
                    looking_down_start_time = None
                    DISPLAY_VIDEO = False

                # --- UI Rendering ---
                status_text = "Looking at PC"
                color = (0, 255, 0)
                
                if not_looking_at_pc:
                    status_text = "Not looking at PC"
                    color = (0, 0, 255)
                elif looking_down_triggered:
                    status_text = "Looking DOWN"
                    color = (255, 0, 0)
                    
                cv2.putText(image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                cv2.putText(image, f"Iris: {avg_ratio:.2f} | Pitch: {head_pitch:.2f}", (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # --- Mambo Video Management (with Audio via ffplay) ---
                if DISPLAY_VIDEO:
                    if ffplay_process is None:
                        # Start ffplay in a separate process
                        # -loop 0: repeat infinitely
                        # -autoexit: exit when video ends (not needed with loop 0 but good practice)
                        # -window_title Mambo: name the window
                        # -noborder: make it look cleaner
                        ffplay_process = subprocess.Popen(
                            ['ffplay', '-loop', '0', '-window_title', 'Mambo', '-noborder', '-alwaysontop', mambo_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                else:
                    if ffplay_process is not None:
                        ffplay_process.terminate()
                        ffplay_process = None

                cv2.imshow('Eye Tracker', image)
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        if ffplay_process:
            ffplay_process.terminate()
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()
