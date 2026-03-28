# Human Face Recognition System

Follows the algorithm in the reference PDF (CSE4021). Free tools only: Python, OpenCV, NumPy,
the YuNet face detector and a pretrained ArcFace model (both downloaded automatically on first run).

| PDF step | Where |
|---|---|
| 2–3 Collect images, one label per person | `collect_faces.py` → `dataset/<LABEL>/` |
| 4 Read image, consistent colour format | `face_system.py` `read_image` |
| 5 Detect faces (YuNet) | `detect_faces` |
| 6 Landmarks → align by eyes → crop → resize 112×112 → normalise | `preprocess` |
| 7 Feature extraction (ArcFace, 512-d) | `extract_features` |
| 8 Store features + labels | `enroll.py` → `face_db.npz` |
| 9–11 Test image / webcam frame, same pipeline | `recognize.py` |
| 12–14 Euclidean distance, minimum, threshold T | `match` |
| 15 Bounding box + name + confidence | `recognize.py` |
| 16–17 Every face, every frame; Q/Esc exits | `recognize.py` |

## Setup
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Use
1. **Collect images** (about 10–20 per person, with different lighting, expressions, angles and backgrounds):
   ```bash
   python collect_faces.py 21BCE1234     # SPACE = save picture, Q = stop
   ```
   Or copy photos into `dataset/21BCE1234/`. The folder name is the identity label.
2. **Build the face database**:
   ```bash
   python enroll.py
   ```
   Run this again whenever you add people or images.
3. **Recognise**:
   ```bash
   python recognize.py                   # live webcam, Q/Esc to exit
   python recognize.py --image test.jpg  # one image, also saves result.jpg
   ```
   `--threshold 1.1` is the default T. Use a lower value (e.g. 1.0) to be stricter.
   Faces with d_min > T are labelled "Unknown Person".
