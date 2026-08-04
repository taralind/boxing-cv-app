"""
pipeline_core.py
─────────────────
Refactored core of boxing_pipeline_withwarp.py for use inside a Streamlit app.

What changed vs. the original script (logic is otherwise identical):
  • No argparse / CLI / multi-round folder discovery — the app works on ONE
    uploaded video at a time (treated as "round 1").
  • No cv2.namedWindow / imshow / waitKey / setMouseCallback anywhere. Every
    interactive routine (label_tracks, select_ring_corners) has been split
    into:
        - a pure "render" function that draws the current state to an image
        - a pure "handle click" function that updates state given an (x, y)
    The Streamlit layer (app.py) owns the actual click widget and loop.
  • All paths are parameterised (work_dir) instead of "./..." so multiple
    Streamlit sessions never collide.
  • Heavy libraries (ultralytics, rtmlib, supervision) are imported lazily
    inside the functions that need them, so the app can start up and render
    the upload screen without them installed/working yet.

Every numeric constant, threshold, and algorithm below is copied verbatim
from the original boxing_pipeline_withwarp.py so pipeline behaviour/outputs
are unchanged.
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Config (verbatim from original script)
# ──────────────────────────────────────────────────────────────────────────────

IDENTITIES  = ["Blue boxer", "Red boxer", "Judge"]
CLASS_NAMES = ["Blue_boxer", "Red_boxer", "Judge"]

ID_COLOURS = {          # BGR
    "Blue_boxer": (200,  80,  20),
    "Red_boxer":  ( 30,  30, 210),
    "Judge":      ( 60, 180,  60),
}
ID_COLOURS_LABEL = {
    "Blue boxer": (200,  80,  20),
    "Red boxer":  ( 30,  30, 210),
    "Judge":      ( 60, 180,  60),
}

SKELETON_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
LEFT_ANKLE_IDX  = 15
RIGHT_ANKLE_IDX = 16
ANKLE_BGR = (0, 255, 255)

RING_WIDTH_CM  = 610
RING_HEIGHT_CM = 610
CANVAS_W = 610
CANVAS_H = 610

CORNER_LABELS = ["top-left", "top-right", "bottom-right", "bottom-left"]
CORNER_COLOUR = (0, 255, 255)

NUM_SEED_FRAMES  = 10
PROP_WINDOW      = 20
RTMPOSE_MODE     = "performance"
RTMPOSE_BACKEND  = "onnxruntime"
KP_CONF_THRESH   = 0.50

MODEL_SIZE   = "yolov8s.pt"
IMG_SIZE     = 640
YOLO_CONF    = 0.60

# Original hard-coded cache path where rtmlib stores the RTMPose-x checkpoint
# after Body() downloads it once during the labelling phase.
RTMPOSE_ONNX = (
    "~/.cache/rtmlib/hub/checkpoints/"
    "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.onnx"
)
RTMPOSE_INPUT_SIZE = (288, 384)

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}


# ──────────────────────────────────────────────────────────────────────────────
# Small data holders
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SeedResult:
    seed_fi: int
    seed_fname: str
    track_ids: list
    bboxes: list          # list of [x1,y1,x2,y2]


@dataclass
class WorkPaths:
    """All on-disk locations for one Streamlit session's pipeline run."""
    root: str

    def __post_init__(self):
        self.frames_dir      = os.path.join(self.root, "frames")
        self.staging_dir     = os.path.join(self.root, "staging")   # labelled frames before split
        self.dataset_dir     = os.path.join(self.root, "dataset")
        self.dataset_yaml    = os.path.join(self.root, "dataset.yaml")
        self.runs_dir        = os.path.join(self.root, "runs")
        self.output_frames   = os.path.join(self.root, "output_frames")
        self.homography_npy  = os.path.join(self.root, "homography.npy")
        self.ankle_csv       = os.path.join(self.root, "ankle_positions.csv")
        self.enriched_csv    = os.path.join(self.root, "ankle_positions_enriched.csv")
        self.output_video    = os.path.join(self.root, "annotated_output.mp4")
        self.uploaded_video  = os.path.join(self.root, "input_video.mp4")
        self.uploaded_weights = os.path.join(self.root, "uploaded_best.pt")
        for d in (self.frames_dir, self.staging_dir, self.output_frames, self.runs_dir):
            os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# ffmpeg — frame extraction / video stitching
# ──────────────────────────────────────────────────────────────────────────────

def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def extract_frames(video_path: str, frames_dir: str, fps: int) -> int:
    """Extract frames from a video into frames_dir at the given fps. Returns count."""
    os.makedirs(frames_dir, exist_ok=True)
    existing = [f for f in os.listdir(frames_dir) if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg"}]
    if existing:
        return len(existing)

    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        "-start_number", "0",
        os.path.join(frames_dir, "%05d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr[-2000:]}")

    return len([f for f in os.listdir(frames_dir) if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg"}])


def stitch_video(frames_dir: str, output_path: str, fps: int) -> str:
    """Stitch numbered jpgs in frames_dir into an mp4 for browser preview/download."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "%05d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg stitching failed (exit {result.returncode}):\n{result.stderr[-2000:]}")
    return output_path


def collect_frames(folder: str) -> list[str]:
    exts = {".jpg", ".jpeg"}
    names = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
    names.sort(key=lambda f: int(os.path.splitext(f)[0]))
    if not names:
        raise FileNotFoundError(f"No JPEG frames found in {folder!r}")
    return names


def sample_seed_indices(n_frames: int, k: int) -> list[int]:
    if k >= n_frames:
        return list(range(n_frames))
    if k == 1:
        return [n_frames // 2]
    return [int(round(i * (n_frames - 1) / (k - 1))) for i in range(k)]


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1a — Detection + tracking on a seed frame (bootstraps identity labels)
# ──────────────────────────────────────────────────────────────────────────────

def kp_to_sv_detections(kp_raw, sc_raw, img_shape, sv):
    if kp_raw is None or len(kp_raw) == 0:
        return sv.Detections.empty()

    kp = np.array(kp_raw, dtype=np.float32)
    sc = np.array(sc_raw, dtype=np.float32)
    if kp.ndim == 2:
        kp, sc = kp[None], sc[None]

    H, W = img_shape[:2]
    bboxes, confs = [], []
    for i in range(kp.shape[0]):
        kps = np.concatenate([kp[i], sc[i, :, None]], axis=1)
        vis = kps[kps[:, 2] > KP_CONF_THRESH]
        if len(vis) < 4:
            continue
        pad = 20
        x1 = max(float(vis[:, 0].min()) - pad, 0)
        y1 = max(float(vis[:, 1].min()) - pad, 0)
        x2 = min(float(vis[:, 0].max()) + pad, W)
        y2 = min(float(vis[:, 1].max()) + pad, H)
        bboxes.append([x1, y1, x2, y2])
        confs.append(float(sc[i].mean()))

    if not bboxes:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(bboxes, dtype=np.float32),
        confidence=np.array(confs, dtype=np.float32),
        class_id=np.zeros(len(bboxes), dtype=int),
    )


def get_bytetrack_cls():
    import supervision as sv
    TrackerCls = getattr(sv, "ByteTrack", None) or getattr(sv, "ByteTracker", None)
    if TrackerCls is None:
        raise RuntimeError("supervision has neither ByteTrack nor ByteTracker — pip install --upgrade supervision")
    return sv, TrackerCls


def make_tracker(TrackerCls):
    return TrackerCls(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=30,
    )


_body_model_cache = {}


def get_body_model(device: str):
    """rtmlib Body = RTMDet (person detector) + RTMPose, used only to bootstrap
    identity labelling on seed frames before a custom YOLO model exists."""
    key = device
    if key not in _body_model_cache:
        from rtmlib import Body
        _body_model_cache[key] = Body(mode=RTMPOSE_MODE, backend=RTMPOSE_BACKEND, device=device)
    return _body_model_cache[key]


def detect_and_track_seed(seed_img: np.ndarray, device: str) -> tuple:
    """Run Body() detector+pose on a seed frame, then ByteTrack, to get track ids/boxes
    for the user to click-label. Returns (tracked_ids, bboxes, window_tracker)."""
    sv, TrackerCls = get_bytetrack_cls()
    body = get_body_model(device)
    window_tracker = make_tracker(TrackerCls)

    kp_raw, sc_raw = body(seed_img)
    sv_dets = kp_to_sv_detections(kp_raw, sc_raw, seed_img.shape, sv)
    tracked = window_tracker.update_with_detections(sv_dets)

    if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
        return [], [], window_tracker
    return tracked.tracker_id.tolist(), tracked.xyxy.tolist(), window_tracker


def propagate_window(window_tracker, frames_dir: str, frame_names: list, seed_fi: int,
                      n_frames: int, device: str) -> list:
    """Continue tracking for PROP_WINDOW frames after a confirmed seed. Returns
    list of (fi, fname, img_bgr, track_ids, bboxes)."""
    sv, _ = get_bytetrack_cls()
    body = get_body_model(device)

    out = []
    end_fi = min(seed_fi + PROP_WINDOW, n_frames)
    for fi in range(seed_fi + 1, end_fi):
        fname = frame_names[fi]
        img = cv2.imread(os.path.join(frames_dir, fname))
        if img is None:
            continue
        kp_raw, sc_raw = body(img)
        sv_dets = kp_to_sv_detections(kp_raw, sc_raw, img.shape, sv)
        tracked = window_tracker.update_with_detections(sv_dets)
        if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
            continue
        out.append((fi, fname, img, tracked.tracker_id.tolist(), tracked.xyxy.tolist()))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1b — Click-to-label UI helpers (replaces cv2 label_tracks GUI)
# ──────────────────────────────────────────────────────────────────────────────

def track_at(cx: float, cy: float, track_ids: list, bboxes: list):
    """Find which track a click landed on: inside a box first, else nearest centre."""
    for tid, (x1, y1, x2, y2) in zip(track_ids, bboxes):
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return tid
    dists = [((cx - (x1 + x2) / 2) ** 2 + (cy - (y1 + y2) / 2) ** 2) for (x1, y1, x2, y2) in bboxes]
    return track_ids[int(np.argmin(dists))]


def render_track_assignment(img: np.ndarray, track_ids: list, bboxes: list, assigned: dict) -> np.ndarray:
    """Draw current click-assignment state onto the seed frame (BGR)."""
    canvas = img.copy()
    for tid, (x1, y1, x2, y2) in zip(track_ids, bboxes):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if tid in assigned:
            ident = assigned[tid]
            colour = ID_COLOURS_LABEL[ident]
            label = f"{ident}  [ID {tid}]"
        else:
            colour = (160, 160, 160)
            label = f"ID {tid}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(canvas, label, (x1 + 4, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x1 + 4, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def build_label_lines(track_ids: list, bboxes: list, identity_map: dict, class_map: dict, W: int, H: int) -> list:
    present = {identity_map[tid] for tid in track_ids if tid in identity_map}
    if len(present) < 3:
        return []
    lines = []
    for tid, (x1, y1, x2, y2) in zip(track_ids, bboxes):
        if tid not in identity_map:
            continue
        ident = identity_map[tid]
        cls = class_map[ident]
        xc = ((x1 + x2) / 2) / W
        yc = ((y1 + y2) / 2) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def stage_annotated_frame(staging_dir: str, fi: int, fname: str, img: np.ndarray, label_lines: list):
    """Write one labelled frame (jpg + yolo txt) to the staging dir, keyed by frame
    index so repeated windows never overwrite an already-staged frame (first wins,
    same dedup semantics as the original script's `seen_fi` set)."""
    if not label_lines:
        return
    stem = os.path.splitext(fname)[0]
    label_path = os.path.join(staging_dir, f"{stem}.txt")
    if os.path.exists(label_path):
        return
    cv2.imwrite(os.path.join(staging_dir, f"{stem}.jpg"), img)
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))


def finalize_dataset(staging_dir: str, dataset_dir: str, dataset_yaml_path: str,
                      class_names: list, seed: Optional[int] = None) -> dict:
    """80/20 shuffle-split every staged (jpg, txt) pair into a YOLO dataset."""
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(staging_dir) if f.endswith(".txt"))
    if not stems:
        raise RuntimeError("No labelled frames were staged — confirm at least one seed before training.")

    if seed is not None:
        random.Random(seed).shuffle(stems)
    else:
        random.shuffle(stems)

    imgs_train = os.path.join(dataset_dir, "images", "train")
    imgs_val   = os.path.join(dataset_dir, "images", "val")
    lbs_train  = os.path.join(dataset_dir, "labels", "train")
    lbs_val    = os.path.join(dataset_dir, "labels", "val")
    for d in (imgs_train, imgs_val, lbs_train, lbs_val):
        os.makedirs(d, exist_ok=True)

    split = max(1, int(len(stems) * 0.8))
    train_stems, val_stems = stems[:split], stems[split:]

    for subset, img_dir, lbl_dir in [(train_stems, imgs_train, lbs_train), (val_stems, imgs_val, lbs_val)]:
        for stem in subset:
            shutil.copy(os.path.join(staging_dir, f"{stem}.jpg"), os.path.join(img_dir, f"{stem}.jpg"))
            shutil.copy(os.path.join(staging_dir, f"{stem}.txt"), os.path.join(lbl_dir, f"{stem}.txt"))

    with open(dataset_yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(dataset_dir)}\n")
        f.write("train: images/train\n")
        f.write("val:   images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    return {"train": len(train_stems), "val": len(val_stems), "total": len(stems)}


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1c — Ring-corner click UI helpers (replaces cv2 select_ring_corners GUI)
# ──────────────────────────────────────────────────────────────────────────────

def render_corner_canvas(frame: np.ndarray, clicks: list) -> np.ndarray:
    canvas = frame.copy()
    for i, (x, y) in enumerate(clicks):
        cv2.circle(canvas, (int(x), int(y)), 8, CORNER_COLOUR, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(x), int(y)), 9, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, CORNER_LABELS[i], (int(x) + 10, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, CORNER_LABELS[i], (int(x) + 10, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CORNER_COLOUR, 1, cv2.LINE_AA)
    for i in range(1, len(clicks)):
        cv2.line(canvas, tuple(map(int, clicks[i - 1])), tuple(map(int, clicks[i])), CORNER_COLOUR, 1, cv2.LINE_AA)
    if len(clicks) == 4:
        cv2.line(canvas, tuple(map(int, clicks[3])), tuple(map(int, clicks[0])), CORNER_COLOUR, 1, cv2.LINE_AA)
    return canvas


def compute_homography(src_corners: np.ndarray) -> np.ndarray:
    dst_corners = np.array([
        [0, 0],
        [CANVAS_W, 0],
        [CANVAS_W, CANVAS_H],
        [0, CANVAS_H],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src_corners, dst_corners)
    return H


def warp_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    warped = (H @ pts_h.T).T
    warped = warped[:, :2] / warped[:, 2:3]
    return warped.astype(np.float32)


def render_warped_preview(frame: np.ndarray, H: np.ndarray) -> np.ndarray:
    warped_frame = cv2.warpPerspective(frame, H, (CANVAS_W, CANVAS_H))
    cv2.rectangle(warped_frame, (0, 0), (CANVAS_W - 1, CANVAS_H - 1), (0, 255, 255), 2)
    cx, cy = CANVAS_W // 2, CANVAS_H // 2
    cv2.drawMarker(warped_frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 30, 1, cv2.LINE_AA)
    return warped_frame


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — YOLO training
# ──────────────────────────────────────────────────────────────────────────────

def train_yolo(dataset_yaml_path: str, epochs: int, batch: int, device: str,
                project_dir: str, progress_cb=None, run_name: str = "custom_boxer_tracker") -> str:
    from ultralytics import YOLO
    model = YOLO(MODEL_SIZE)

    if progress_cb is not None:
        def on_train_epoch_end(trainer):
            # trainer.epoch is 0-indexed, so we add 1
            progress_cb(trainer.epoch + 1, trainer.epochs)
            
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        
    model.train(
        data=os.path.abspath(dataset_yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=IMG_SIZE,
        device=device,
        project=project_dir,
        name=run_name,
        exist_ok=True,
    )
    weights_path = os.path.join(project_dir, run_name, "weights", "best.pt")
    if not os.path.exists(weights_path):
        raise RuntimeError(f"Training finished but weights not found at {weights_path}")
    return weights_path


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — YOLO + RTMPose inference
# ──────────────────────────────────────────────────────────────────────────────

def load_yolo_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


def load_pose_model(device: str):
    from rtmlib import RTMPose
    onnx_path = os.path.expanduser(RTMPOSE_ONNX)
    if not os.path.exists(onnx_path):
        # Fall back to letting rtmlib fetch the same checkpoint fresh if the
        # labelling phase (which normally caches it via Body()) wasn't run
        # in this session — e.g. the user uploaded pre-trained YOLO weights
        # and skipped straight to inference.
        get_body_model(device)
    return RTMPose(
        onnx_model=onnx_path,
        model_input_size=RTMPOSE_INPUT_SIZE,
        backend=RTMPOSE_BACKEND,
        device=device,
    )


def ankle_record(kps: np.ndarray) -> dict:
    out = {}
    for side, idx in (("left_ankle", LEFT_ANKLE_IDX), ("right_ankle", RIGHT_ANKLE_IDX)):
        x, y, conf = kps[idx]
        if conf > KP_CONF_THRESH:
            out[side] = {"x": round(float(x), 2), "y": round(float(y), 2), "confidence": round(float(conf), 4)}
        else:
            out[side] = None
    return out


def draw_pose(img: np.ndarray, kps: np.ndarray, colour) -> np.ndarray:
    for a, b in SKELETON_PAIRS:
        xa, ya, ca = kps[a]; xb, yb, cb = kps[b]
        if ca > KP_CONF_THRESH and cb > KP_CONF_THRESH:
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), colour, 2, cv2.LINE_AA)
    for x, y, conf in kps:
        if conf > KP_CONF_THRESH:
            cv2.circle(img, (int(x), int(y)), 4, colour, -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 1, cv2.LINE_AA)
    for ai in (LEFT_ANKLE_IDX, RIGHT_ANKLE_IDX):
        x, y, conf = kps[ai]
        if conf > KP_CONF_THRESH:
            cv2.circle(img, (int(x), int(y)), 9, ANKLE_BGR, 2, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), 4, ANKLE_BGR, -1, cv2.LINE_AA)
    return img


def draw_label_box(img: np.ndarray, bbox, label: str, colour) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(img, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, y1), colour, -1)
    cv2.putText(img, label, (x1 + 3, max(y1 - 4, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def run_inference_frame(img: np.ndarray, fi: int, yolo_model, pose_model) -> tuple:
    """Run YOLO identity detection + RTMPose top-down on one frame, draw overlays,
    and return (annotated_img_bgr, frame_record dict)."""
    frame_record = {}

    yolo_out = yolo_model(img, verbose=False, conf=YOLO_CONF)[0]
    yolo_boxes = {}
    for box in yolo_out.boxes:
        cls_id = int(box.cls[0])
        ident_name = yolo_model.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        yolo_boxes[ident_name] = [x1, y1, x2, y2]

    for ident in CLASS_NAMES:
        colour = ID_COLOURS[ident]
        if ident in yolo_boxes:
            bbox = yolo_boxes[ident]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            kp_raw, sc_raw = pose_model(img, bboxes=np.array([[x1, y1, x2, y2]]))
            if kp_raw is not None and len(kp_raw) > 0:
                kp = np.array(kp_raw[0], dtype=np.float32)
                sc = np.array(sc_raw[0], dtype=np.float32)
                kps = np.concatenate([kp, sc[:, None]], axis=1)
                draw_pose(img, kps, colour)
                draw_label_box(img, bbox, ident, colour)
                frame_record[ident] = {
                    "status": "detected", "bbox": [float(v) for v in bbox],
                    "keypoints": kps.tolist(), "ankles": ankle_record(kps),
                }
            else:
                draw_label_box(img, bbox, f"{ident} [No Pose]", colour)
                frame_record[ident] = {"status": "box_only", "bbox": [float(v) for v in bbox],
                                        "keypoints": None, "ankles": None}
        else:
            frame_record[ident] = {"status": "lost", "bbox": None, "keypoints": None, "ankles": None}

    cv2.putText(img, f"frame {fi}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, f"frame {fi}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    return img, frame_record


def rows_from_results(all_results: dict, round_num: int, match_name: str) -> tuple:
    rows = []
    for fi, data in sorted(all_results.items()):
        for ident, rec in data["people"].items():
            la = (rec["ankles"] or {}).get("left_ankle") or {}
            ra = (rec["ankles"] or {}).get("right_ankle") or {}
            rows.append([
                round_num, match_name, fi, data["frame"], ident, rec["status"],
                la.get("x", ""), la.get("y", ""), la.get("confidence", ""),
                ra.get("x", ""), ra.get("y", ""), ra.get("confidence", ""),
            ])
    header = [
        "Round", "match_name", "frame_idx", "frame_file", "identity", "status",
        "left_ankle_x", "left_ankle_y", "left_ankle_conf",
        "right_ankle_x", "right_ankle_y", "right_ankle_conf",
    ]
    return header, rows


def write_csv(path: str, header: list, rows: list):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 — Homography transform + interpolation of the ankle CSV
# ──────────────────────────────────────────────────────────────────────────────

ANKLE_COORD_COLS = ["left_ankle_x", "left_ankle_y", "right_ankle_x", "right_ankle_y"]


def interpolate_lost(df: pd.DataFrame) -> pd.DataFrame:
    filled_parts = []
    for identity, group in df.groupby("identity"):
        g = group.sort_values("frame_idx").copy()
        lost_mask = g["status"] == "lost"
        g.loc[lost_mask, ANKLE_COORD_COLS] = np.nan
        g[ANKLE_COORD_COLS] = g[ANKLE_COORD_COLS].interpolate(method="linear", limit_direction="both")
        newly_filled = lost_mask & g[ANKLE_COORD_COLS].notna().any(axis=1)
        g.loc[newly_filled, "status"] = "interpolated"
        filled_parts.append(g)
    if not filled_parts:
        return df.iloc[0:0]
    return pd.concat(filled_parts).sort_values(["frame_idx", "identity"]).reset_index(drop=True)


def warp_and_enrich(df: pd.DataFrame, H: np.ndarray) -> pd.DataFrame:
    """Apply homography H to boxer ankle coordinates; adds warped/relative/distance
    columns; keeps only rows where at least one ankle lands inside the ring canvas."""
    df = df[df["identity"].isin(["Blue_boxer", "Red_boxer"])].copy()
    df = interpolate_lost(df)
    if df.empty:
        return df

    CENTRE_X = CANVAS_W / 2
    CENTRE_Y = CANVAS_H / 2

    def _warp_side(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.copy()
        for side in ("left", "right"):
            x_col, y_col = f"{side}_ankle_x", f"{side}_ankle_y"
            n = len(sub)
            warped_x = np.full(n, np.nan); warped_y = np.full(n, np.nan)
            rel_x = np.full(n, np.nan); rel_y = np.full(n, np.nan)
            dist_cm = np.full(n, np.nan)

            valid_mask = sub[x_col].notna() & sub[y_col].notna()
            if valid_mask.any():
                raw_xy = sub.loc[valid_mask, [x_col, y_col]].values
                warped = warp_points(raw_xy, H)
                idx = np.where(valid_mask)[0]
                warped_x[idx] = warped[:, 0]
                warped_y[idx] = CANVAS_H - warped[:, 1]
                rel_x[idx] = warped[:, 0] - CENTRE_X
                rel_y[idx] = (CANVAS_H - warped[:, 1]) - CENTRE_Y
                dist_cm[idx] = np.sqrt(rel_x[idx] ** 2 + rel_y[idx] ** 2)

            sub[f"{side}_ankle_warped_x"] = warped_x
            sub[f"{side}_ankle_warped_y"] = warped_y
            sub[f"{side}_ankle_rel_x"] = rel_x
            sub[f"{side}_ankle_rel_y"] = rel_y
            sub[f"{side}_ankle_dist_cm"] = dist_cm

        has_valid = (
            sub["left_ankle_warped_x"].between(0, CANVAS_W) |
            sub["right_ankle_warped_x"].between(0, CANVAS_W)
        )
        return sub[has_valid].copy()

    parts = []
    for identity in ["Blue_boxer", "Red_boxer"]:
        sub_df = df[df["identity"] == identity]
        if not sub_df.empty:
            parts.append(_warp_side(sub_df))

    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts).sort_values(["frame_idx", "identity"]).reset_index(drop=True)
