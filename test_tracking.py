import cv2
import json
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO_PATH = "data/videos/sample.mp4"
MODEL_PATH = "models/best.pt"

OUT_VIDEO = "outputs/videos/tracking_output.mp4"
OUT_JSON = "outputs/json/counts.json"

CONF_THRESH = 0.4
IOU_THRESH = 0.4
FRAME_SKIP = 3
# ----------------------------------------

# Load poultry-trained YOLO model
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open input video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (width, height))

frame_id = 0
counts_over_time = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # Skip frames for speed
    if frame_id % FRAME_SKIP != 0:
        out.write(frame)
        continue

    timestamp = round(frame_id / fps, 2)

    results = model.track(
        frame,
        persist=True,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        verbose=False
        # If your model is multi-class and chicken = 0, add: classes=[0]
    )

    active_ids = set()
    total_weight_index = 0

    for r in results:
        if r.boxes is None or r.boxes.id is None:
            continue

        for box, track_id in zip(r.boxes.xyxy, r.boxes.id):
            x1, y1, x2, y2 = map(int, box)
            tid = int(track_id)

            # ---- Weight proxy (normalized bbox area) ----
            area = (x2 - x1) * (y2 - y1)
            weight_index = area / (width * height)
            total_weight_index += weight_index
            # --------------------------------------------

            active_ids.add(tid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {tid}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    count = len(active_ids)
    avg_weight_index = round(total_weight_index / count, 4) if count > 0 else 0

    counts_over_time.append({
        "time_sec": timestamp,
        "count": count,
        "avg_weight_index": avg_weight_index
    })

    # Overlay count and weight on video
    cv2.putText(
        frame,
        f"Count: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Avg Weight Index: {avg_weight_index}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    out.write(frame)

cap.release()
out.release()

with open(OUT_JSON, "w") as f:
    json.dump({
        "fps": fps,
        "frame_skip": FRAME_SKIP,
        "counts": counts_over_time
    }, f, indent=2)

print("✅ Video saved to:", OUT_VIDEO)
print("✅ JSON saved to:", OUT_JSON)
