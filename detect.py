from ultralytics import YOLO
import cv2
import time
import csv
from datetime import datetime

# Load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)  # webcam

# Line position
LINE_Y = 300

# Person class only
PERSON_CLASS = 0

# Track previous positions
prev_positions = {}

# Counted IDs
counted_in = set()
counted_out = set()

# Counters
people_in = 0
people_out = 0

# CSV setup
csv_file = open("counts.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "people_in", "people_out"])

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.5,
        classes=[PERSON_CLASS]  # ONLY PEOPLE
    )

    # Draw line
    cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (0, 0, 255), 2)

    if results[0].boxes.id is not None:
        for box in results[0].boxes:
            track_id = int(box.id[0])

            x1, y1, x2, y2 = box.xyxy[0]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if track_id in prev_positions:
                prev_y = prev_positions[track_id]

                # IN
                if prev_y < LINE_Y and cy > LINE_Y and track_id not in counted_in:
                    people_in += 1
                    counted_in.add(track_id)
                    csv_writer.writerow([datetime.now(), people_in, people_out])

                # OUT
                elif prev_y > LINE_Y and cy < LINE_Y and track_id not in counted_out:
                    people_out += 1
                    counted_out.add(track_id)
                    csv_writer.writerow([datetime.now(), people_in, people_out])

            prev_positions[track_id] = cy
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    # Display counters
    cv2.putText(frame, f"People IN: {people_in}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"People OUT: {people_out}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    annotated_frame = results[0].plot()
    cv2.imshow("People Flow Analytics", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
