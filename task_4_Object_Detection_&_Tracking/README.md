# Task 4: Object Detection and Tracking

Real-time object detection and tracking using **OpenCV** for video I/O and a
pre-trained **YOLOv8** model (via the `ultralytics` package) for detection.
Tracking is handled by the **ByteTrack** algorithm that ships built into
`ultralytics`, which assigns a persistent ID to each object as it moves
across frames.

## Features
- Works with a live webcam or a video file
- Pre-trained YOLOv8 model (80 COCO classes: person, car, dog, etc.) —
  auto-downloads the weights on first run
- Draws bounding boxes, class labels, and tracking IDs on every frame
- Optional: save the annotated video to disk
- Optional: run headless (no display window), useful on a server

## Setup

```bash
cd Task_4_Object_Detection
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Note: On first run, `ultralytics` will automatically download the
> `yolov8n.pt` weights file (~6 MB). This requires an internet connection
> once; after that it's cached locally.

## Run

**Webcam (default camera):**
```bash
python app.py
```

**A specific camera index:**
```bash
python app.py --source 1
```

**A video file:**
```bash
python app.py --source path/to/video.mp4
```

**Save the annotated output:**
```bash
python app.py --source video.mp4 --save output.mp4
```

**Headless mode (no preview window, e.g. on a server):**
```bash
python app.py --source video.mp4 --save output.mp4 --no-display
```

Press `q` in the preview window to stop.

## Customizing
- `--model`: swap in a larger/more accurate YOLOv8 variant
  (`yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`) if you need higher
  accuracy and have the GPU/CPU budget for it.
- `--conf`: raise or lower the detection confidence threshold.
- If you'd rather use **Faster R-CNN** or a **Deep SORT** tracker instead of
  YOLO + ByteTrack, the overall pipeline (capture frame → detect → track →
  draw → display/save) stays the same — only the model-loading and
  `model.track(...)` call in `app.py` would need to be replaced with the
  equivalent detector/tracker API calls.
