"""
MILESTONE 1 - Visual Processing & Safety Detection
----------------------------------------------------
Detect PPE violations in images and videos using YOLOv8.
"""
import os
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
DEFAULT_CONF = 0.4

def load_model():
    """Load the trained PPE detection model."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Model loading error: {e}")
        return None

def is_violation(class_name):
    """Classes beginning with 'no' represent PPE violations."""
    name = str(class_name).strip().lower()
    return name.startswith(("no_", "no-", "no "))

def detect(model, image, conf=DEFAULT_CONF):
    """Run YOLO detection on an image or OpenCV frame."""
    if model is None:
        raise ValueError("YOLO model is not loaded.")

    results = model.predict(image, conf=conf, verbose=False)
    if not results or results[0].boxes is None:
        return []

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        name = model.names[class_id]
        confidence = float(box.conf[0])
        coordinates = [int(v) for v in box.xyxy[0]]
        detections.append({
            "class": name,
            "conf": round(confidence, 2),
            "box": coordinates,
            "violation": is_violation(name),
        })
    return detections

def draw(image, detections):
    """Draw green boxes for PPE and red boxes for violations."""
    output = image.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = (0, 0, 255) if d["violation"] else (0, 200, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{d['class']} {d['conf']:.2f}",
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return output

def summarize(detections, source="image"):
    """Create a structured summary for the dashboard and RAG agent."""
    counts = {}
    violations = []
    for d in detections:
        name = d["class"]
        counts[name] = counts.get(name, 0) + 1
        if d["violation"]:
            violations.append(name)
    violations = sorted(set(violations))

    if violations:
        status = "non-compliant"
        compliant = False
        text = (
            f"{source}: found {counts}. "
            f"VIOLATIONS: {violations}."
        )
    elif detections:
        status = "no violation detected"
        compliant = True
        text = f"{source}: found {counts}. No PPE violations detected."
    else:
        status = "unknown"
        compliant = None
        text = f"{source}: nothing detected. Compliance cannot be determined."

    return {
        "text": text,
        "counts": counts,
        "violations": violations,
        "compliant": compliant,
        "compliance_status": status,
    }

def detect_video(model, video_path, every_n=5, conf=DEFAULT_CONF):
    """
    Analyse every Nth video frame.
    This aggregates sampled-frame detections; it does not track
    individual workers across frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    best = {}
    preview = None
    most_violations = -1
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % every_n != 0:
            continue
        detections = detect(model, frame, conf)
        # Keep the busiest sampled frame for each class.
        per_class = {}
        for d in detections:
            per_class.setdefault(d["class"], []).append(d)

        for name, items in per_class.items():
            if len(items) > len(best.get(name, [])):
                best[name] = items

        # Save the frame containing the most violations.
        violation_count = sum(
            d["violation"] for d in detections
        )
        if violation_count > most_violations:
            most_violations = violation_count
            preview = draw(frame, detections)
    cap.release()
    detections = [
        d
        for items in best.values()
        for d in items
    ]
    return detections, preview, frame_no

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python vision.py path/to/image.jpg")
        raise SystemExit(1)

    model = load_model()
    if model is None:
        print(f"best.pt not found: {MODEL_PATH}")
        raise SystemExit(1)
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        raise SystemExit(1)
    detections = detect(model, image)
    summary = summarize(detections, os.path.basename(image_path))
    print(summary["text"])
    print("Compliance:", summary["compliance_status"])
    cv2.imwrite(
        "output.jpg",
        draw(image, detections)
    )
    print("Saved output.jpg")