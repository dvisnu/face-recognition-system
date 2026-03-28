"""Steps 2-3: collect face images of one person from the webcam.

Usage: python collect_faces.py <IDENTITY_LABEL> [--count 20]
Images are saved to dataset/<IDENTITY_LABEL>/. Change lighting, expression,
head angle and background while capturing.
Press SPACE to save a picture, Q to finish.
"""
import argparse
import os

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("label", help="unique identity label, e.g. Reg_No / Roll_No / EID")
parser.add_argument("--count", type=int, default=20, help="number of images to capture")
parser.add_argument("--camera", type=int, default=0)
args = parser.parse_args()

folder = os.path.join("dataset", args.label)
os.makedirs(folder, exist_ok=True)
cam = cv2.VideoCapture(args.camera)
if not cam.isOpened():
    raise SystemExit("Could not open webcam. You can also copy photos into " + folder)

saved = len(os.listdir(folder))
target = saved + args.count
while saved < target:
    ok, frame = cam.read()
    if not ok:
        break
    view = frame.copy()
    cv2.putText(view, f"{args.label}: {saved}/{target}  SPACE=save  Q=quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Collect faces", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        cv2.imwrite(os.path.join(folder, f"{saved:03d}.jpg"), frame)
        saved += 1
    elif key in (ord("q"), 27):
        break

cam.release()
cv2.destroyAllWindows()
print(f"{saved} images in {folder}")
