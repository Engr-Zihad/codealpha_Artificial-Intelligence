"""
Task 4: Object Detection and Tracking
--------------------------------------
Real-time object detection + tracking using:
  - OpenCV for video capture (webcam or file) and drawing
  - A pre-trained YOLO model (via the `ultralytics` package, YOLOv8) for detection
  - Built-in ByteTrack tracker (bundled with ultralytics) for object tracking,
    which assigns a persistent ID to each tracked object across frames

Optional extra feature:
  - Age & gender estimation per tracked person, using the `deepface` library.
    This only runs when person-class objects are detected and --age-gender
    is passed. Age/gender models are pre-trained and approximate, not exact.

Usage:
    # Webcam (default camera index 0)
    python app.py

    # A specific camera index
    python app.py --source 1

    # A video file
    python app.py --source path/to/video.mp4

    # Save the annotated output to a file instead of / as well as showing it
    python app.py --source video.mp4 --save output.mp4

    # Also estimate age & gender for each detected person
    python app.py --source video.mp4 --save output.mp4 --age-gender

Press 'q' to quit the live window.
"""

import argparse
import cv2
from ultralytics import YOLO

# COCO class id for "person" (used to filter detections when --age-gender is on)
PERSON_CLASS_ID = 0

# What fraction of a person's bounding box (from the top) we treat as the
# "head/face" region when cropping for age/gender analysis. DeepFace works
# far better on a face crop than on a full-body crop, so we only send the
# top portion of each person box instead of the whole body.
HEAD_CROP_FRACTION = 0.35

# How often (in frames) to re-run age/gender analysis per tracked ID.
# Running it every single frame is slow and unnecessary since a person's
# estimated age/gender won't change frame-to-frame.
AGE_GENDER_REFRESH_INTERVAL = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking")
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: webcam index (e.g. 0) or path to a video file (default: 0)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path or name of the YOLO model weights to use (default: yolov8n.pt, "
        "auto-downloaded on first run).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Minimum confidence threshold for detections (default: 0.4)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the annotated video output (e.g. output.mp4)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't open a live preview window (useful for headless/server runs)",
    )
    parser.add_argument(
        "--age-gender",
        action="store_true",
        help="Also estimate age & gender for each detected person using DeepFace "
        "(requires: pip install deepface tf-keras). This restricts detection to "
        "the 'person' class and draws custom labels instead of default YOLO labels.",
    )
    return parser.parse_args()


def resolve_source(source: str):
    """Webcam indices come in as strings ('0'); convert to int if numeric."""
    return int(source) if source.isdigit() else source


def estimate_age_gender(deepface_module, frame, x1, y1, x2, y2):
    """Crop the head/face region of a person box and run DeepFace age/gender
    analysis on it. Returns a label string, or None if analysis fails
    (e.g. no clear face visible)."""
    box_height = y2 - y1
    head_y2 = y1 + int(box_height * HEAD_CROP_FRACTION)
    face_crop = frame[max(0, y1):head_y2, max(0, x1):x2]

    if face_crop.size == 0:
        return None

    try:
        analysis = deepface_module.DeepFace.analyze(
            face_crop, actions=["age", "gender"],
            enforce_detection=False, silent=True,
        )
        result = analysis[0]
        age = int(result["age"])

        # The gender result key differs slightly across deepface versions,
        # so check both possible shapes.
        if "dominant_gender" in result:
            gender = result["dominant_gender"]
        elif isinstance(result.get("gender"), dict):
            gender = max(result["gender"], key=result["gender"].get)
        else:
            gender = "Unknown"

        return f"{gender}, ~{age}y"
    except Exception:
        return None


def main():
    args = parse_args()
    source = resolve_source(args.source)

    # Load a pre-trained YOLOv8 model (COCO-trained by default: 80 common object classes)
    model = YOLO(args.model)

    deepface_module = None
    if args.age_gender:
        try:
            import deepface as deepface_module
        except ImportError:
            raise ImportError(
                "Age/gender estimation requires the 'deepface' package. "
                "Install it with: pip install deepface tf-keras"
            )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))

    print("Starting detection + tracking. Press 'q' to quit.")

    # Remembers the last known age/gender label per track ID, so labels
    # don't flicker between frames where we skip re-analysis.
    last_known_label = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream / cannot read frame.")
            break
        frame_count += 1

        # model.track runs detection AND assigns persistent tracker IDs
        # (ByteTrack, bundled with ultralytics) in one call.
        track_kwargs = dict(conf=args.conf, persist=True, verbose=False)
        if args.age_gender:
            track_kwargs["classes"] = [PERSON_CLASS_ID]  # only track people

        results = model.track(frame, **track_kwargs)

        if args.age_gender:
            # Custom drawing path: boxes + age/gender labels
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    track_id = int(box.id[0]) if box.id is not None else -1

                    needs_refresh = (
                        frame_count % AGE_GENDER_REFRESH_INTERVAL == 0
                        or track_id not in last_known_label
                    )
                    if needs_refresh:
                        label_suffix = estimate_age_gender(deepface_module, frame, x1, y1, x2, y2)
                        last_known_label[track_id] = (
                            f"ID {track_id} | {label_suffix}" if label_suffix
                            else f"ID {track_id} | Face not clear"
                        )

                    label = last_known_label.get(track_id, f"ID {track_id}")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            annotated_frame = frame
        else:
            # Default path: let ultralytics draw boxes, class labels, and track IDs
            annotated_frame = frame
            if results and len(results) > 0:
                annotated_frame = results[0].plot()

        if writer is not None:
            writer.write(annotated_frame)

        if not args.no_display:
            cv2.imshow("Object Detection & Tracking (press 'q' to quit)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
