"""
Task 4: Object Detection and Tracking
--------------------------------------
Real-time object detection + tracking using:
  - OpenCV for video capture (webcam or file) and drawing
  - A pre-trained YOLO model (via the `ultralytics` package, YOLOv8) for detection
  - Built-in ByteTrack tracker (bundled with ultralytics) for object tracking,
    which assigns a persistent ID to each tracked object across frames

Usage:
    # Webcam (default camera index 0)
    python app.py

    # A specific camera index
    python app.py --source 1

    # A video file
    python app.py --source path/to/video.mp4

    # Save the annotated output to a file instead of / as well as showing it
    python app.py --source video.mp4 --save output.mp4

Press 'q' to quit the live window.
"""

import argparse
import cv2
from ultralytics import YOLO


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
    return parser.parse_args()


def resolve_source(source: str):
    """Webcam indices come in as strings ('0'); convert to int if numeric."""
    return int(source) if source.isdigit() else source


def main():
    args = parse_args()
    source = resolve_source(args.source)

    # Load a pre-trained YOLOv8 model (COCO-trained by default: 80 common object classes)
    model = YOLO(args.model)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream / cannot read frame.")
            break

        # model.track runs detection AND assigns persistent tracker IDs
        # (ByteTrack, bundled with ultralytics) in one call.
        results = model.track(
            frame,
            conf=args.conf,
            persist=True,   # remember track IDs between frames
            verbose=False,
        )

        annotated_frame = frame
        if results and len(results) > 0:
            result = results[0]
            annotated_frame = result.plot()  # draws boxes, class labels, and track IDs

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
