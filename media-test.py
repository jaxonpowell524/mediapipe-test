import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import os
import csv

os.environ["QT_QPA_PLATFORM"] = "offscreen"

model_path = 'pose_landmarker_full.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

video_filepath = 'preprocessed_mp4_videos/IMG_7086.mp4'


mp4Path = Path(video_filepath)
csvPath = mp4Path.with_suffix("")
csvPath = Path("landmark_csv/" + csvPath.name + "-landmarks.csv")

mp4Path = str(mp4Path)
csvPath = str(csvPath)

cap = cv2.VideoCapture(mp4Path)
landmarks_filepath = csvPath

csv_file = open(landmarks_filepath, mode="w", newline="")
csv_writer = csv.writer(csv_file)

# header (frame index + 33 landmarks × 4 values)
header = ["frame"]
for i in range(33):
    header += [f"x{i}", f"y{i}", f"z{i}", f"vis{i}"]
csv_writer.writerow(header)

frame_idx = 0

with PoseLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  
  while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use current timestamp in ms
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run the pose detection
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw results (if any)
        if result.pose_landmarks:
            for landmark in result.pose_landmarks[0]:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                row = [frame_idx]
                for lm in result.pose_landmarks[0]:
                    row += [lm.x, lm.y, lm.z, lm.visibility]
                csv_writer.writerow(row)

            frame_idx += 1

        #cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

landmarks = [(lm.x, lm.y, lm.z) for lm in result.pose_landmarks[0]]

cap.release()
csv_file.close()
#cv2.destroyAllWindows()