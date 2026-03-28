"""Shared face pipeline: detection, alignment/preprocessing, feature extraction, matching.

Detector : YuNet (OpenCV)  -> face box + 5 landmarks (eyes, nose, mouth corners)
Features : ArcFace (InsightFace w600k_mbf) -> 512-d feature vector
Distance : Euclidean distance between L2-normalised feature vectors
"""
import os
import urllib.request
import zipfile

import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DETECTOR_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
ARCFACE_PATH = os.path.join(MODEL_DIR, "w600k_mbf.onnx")
DETECTOR_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_detection_yunet/face_detection_yunet_2023mar.onnx")
ARCFACE_ZIP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"

FACE_SIZE = 112  # standard size of the aligned face (ArcFace input)
# Where the 5 landmarks should land in the 112x112 aligned face (standard ArcFace template).
TEMPLATE = np.array([[38.2946, 51.6963],   # left eye (in image)
                     [73.5318, 51.5014],   # right eye (in image)
                     [56.0252, 71.7366],   # nose tip
                     [41.5493, 92.3655],   # left mouth corner
                     [70.7299, 92.2041]],  # right mouth corner
                    dtype=np.float32)


def download_models():
    """Download the free pretrained models once (on first run)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(DETECTOR_PATH):
        print("Downloading face detector (YuNet)...")
        urllib.request.urlretrieve(DETECTOR_URL, DETECTOR_PATH)
    if not os.path.exists(ARCFACE_PATH):
        print("Downloading ArcFace model...")
        zip_path = os.path.join(MODEL_DIR, "buffalo_sc.zip")
        urllib.request.urlretrieve(ARCFACE_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            with open(ARCFACE_PATH, "wb") as f:
                f.write(z.read("w600k_mbf.onnx"))
        os.remove(zip_path)


class FaceSystem:
    def __init__(self):
        download_models()
        self.detector = cv2.FaceDetectorYN.create(DETECTOR_PATH, "", (320, 320), 0.8)
        self.arcface = cv2.dnn.readNetFromONNX(ARCFACE_PATH)

    # Step 4: read image into a consistent colour format (3-channel BGR, OpenCV default)
    @staticmethod
    def read_image(path, max_side=1280):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        scale = max_side / max(img.shape[:2])
        if scale < 1:  # shrink very large photos to keep detection fast
            img = cv2.resize(img, None, fx=scale, fy=scale)
        return img

    # Step 5: detect faces. Each row: x, y, w, h, 5 landmarks (x, y), score
    def detect_faces(self, img):
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        return [] if faces is None else list(faces)

    # Step 6: landmarks -> align (using eye coordinates) -> crop -> resize -> normalise
    @staticmethod
    def preprocess(img, face):
        landmarks = face[4:14].reshape(5, 2).astype(np.float32)  # eyes, nose, mouth corners
        # Similarity transform (rotation + scale + shift) that levels the eyes and
        # moves the landmarks to the template positions.
        matrix, _ = cv2.estimateAffinePartial2D(landmarks, TEMPLATE, method=cv2.LMEDS)
        # warpAffine aligns, crops the face region and resizes it to 112x112 in one step
        aligned = cv2.warpAffine(img, matrix, (FACE_SIZE, FACE_SIZE))
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) - 127.5) / 127.5  # pixel values -> [-1, 1]
        return normalized

    # Step 7 / 11: extract the feature vector with ArcFace
    def extract_features(self, normalized_face):
        blob = normalized_face.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
        self.arcface.setInput(blob)
        feature = self.arcface.forward().flatten()
        return feature / np.linalg.norm(feature)  # unit length, so distances are comparable

    def face_to_feature(self, img, face):
        return self.extract_features(self.preprocess(img, face))


# Step 12: Euclidean distance  d_i = sqrt(sum_j (q_j - f_ij)^2)
def euclidean_distance(q, f):
    return float(np.sqrt(np.sum((q - f) ** 2)))


# Steps 12-14: compare with every stored face, take the minimum, apply threshold T
def match(test_feature, db_features, db_labels, threshold):
    min_distance = float("inf")
    predicted = "Unknown Person"
    for feature, label in zip(db_features, db_labels):
        d = euclidean_distance(test_feature, feature)
        if d < min_distance:
            min_distance = d
            predicted = label
    if min_distance <= threshold:
        return predicted, min_distance
    return "Unknown Person", min_distance


def confidence(distance):
    """Confidence score = cosine similarity of the unit vectors (d^2 = 2 - 2cos), as %."""
    return max(0.0, 1 - distance ** 2 / 2) * 100
