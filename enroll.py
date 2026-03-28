"""Steps 4-8: build the face database from dataset/<IDENTITY_LABEL>/*.jpg

Usage: python enroll.py [--dataset dataset] [--db face_db.npz]
"""
import argparse
import os

import numpy as np

from face_system import FaceSystem

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="dataset")
parser.add_argument("--db", default="face_db.npz")
args = parser.parse_args()

system = FaceSystem()
features, labels = [], []  # the face database

for label in sorted(os.listdir(args.dataset)):          # Step 3: folder name = identity label
    person_dir = os.path.join(args.dataset, label)
    if not os.path.isdir(person_dir):
        continue
    count = 0
    for name in sorted(os.listdir(person_dir)):
        img = system.read_image(os.path.join(person_dir, name))  # Step 4
        if img is None:
            continue
        faces = system.detect_faces(img)                          # Step 5
        if not faces:
            print(f"  no face found in {label}/{name}, skipped")
            continue
        # A training photo belongs to one person, so use the largest face in it
        face = max(faces, key=lambda f: f[2] * f[3])
        features.append(system.face_to_feature(img, face))       # Steps 6-7
        labels.append(label)                                      # Step 8
        count += 1
    print(f"{label}: {count} face(s) enrolled")

if not features:
    raise SystemExit("No faces enrolled. Put images in dataset/<label>/ first.")
np.savez(args.db, features=np.array(features), labels=np.array(labels))  # Step 8
print(f"Saved {len(features)} feature vectors for {len(set(labels))} people to {args.db}")
