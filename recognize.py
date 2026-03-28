"""Steps 9-17: recognise faces in a test image or live webcam video.

Usage:
  python recognize.py                      # webcam (press Q or Esc to exit)
  python recognize.py --image test.jpg     # single test image (result also saved)
Options: --threshold T (default 1.1, lower = stricter), --db face_db.npz
"""
import argparse

import cv2
import numpy as np

from face_system import FaceSystem, confidence, match

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="test image path; omit to use the webcam")
parser.add_argument("--db", default="face_db.npz")
parser.add_argument("--threshold", type=float, default=1.1, help="recognition threshold T")
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--no-window", action="store_true", help="image mode: only save result.jpg")
args = parser.parse_args()

system = FaceSystem()
db = np.load(args.db)
db_features, db_labels = db["features"], db["labels"]


def recognize_frame(frame, results=None):
    results = [] if results is None else results
    for face in system.detect_faces(frame):                   # Steps 10, 16: every face
        feature = system.face_to_feature(frame, face)         # Steps 10-11
        name, dist = match(feature, db_features, db_labels, args.threshold)  # Steps 12-14
        # Step 15: bounding box + identity + confidence score
        x, y, w, h = face[:4].astype(int)
        color = (0, 200, 0) if name != "Unknown Person" else (0, 0, 255)
        text = f"{name} ({confidence(dist):.0f}%)" if name != "Unknown Person" else name
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        results.append((name, dist))
    return frame


if args.image:                                                # Step 9: test image
    frame = system.read_image(args.image)
    if frame is None:
        raise SystemExit(f"Could not read {args.image}")
    results = []
    result = recognize_frame(frame, results)
    for name, dist in results:
        print(f"{name:20s} distance={dist:.3f} confidence={confidence(dist):.0f}%")
    cv2.imwrite("result.jpg", result)
    print("Result saved to result.jpg")
    if not args.no_window:
        cv2.imshow("Face Recognition", result)
        cv2.waitKey(0)
else:                                                         # Step 9: video camera
    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        raise SystemExit("Could not open webcam. Use --image <file> instead.")
    while True:                                               # Step 16: every frame
        ok, frame = cam.read()
        if not ok:
            break
        cv2.imshow("Face Recognition (Q to quit)", recognize_frame(frame))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):           # Step 17: user exits
            break
    cam.release()
cv2.destroyAllWindows()
