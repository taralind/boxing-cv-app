"""
Boxing Position Tracking — Streamlit App
──────────────────────────────────────
Upload a boxing match video → manual selection steps (label Blue/Red boxer + Judge on
a handful of seed frames, click the 4 ring corners) → the CV pipeline trains
a detector, runs YOLO + RTMPose inference on every frame, and produces:

  • An annotated video (frames with pose/ankle overlays stitched together)
    so you can validate the tracking worked.
  • A ring-warped CSV of boxer ankle positions that you can download.

Run with:  streamlit run app.py
"""

import os
import shutil
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

import pipeline_core as pc

st.set_page_config(page_title="Boxing Position Tracker", page_icon="🥊", layout="wide")

STAGES = ["upload", "label", "corners", "train", "infer", "results"]
STAGE_LABELS = {
    "upload": "1 · Upload",
    "label": "2 · Label boxers",
    "corners": "3 · Ring corners",
    "train": "4 · Train model",
    "infer": "5 · Run inference",
    "results": "6 · Results",
}


# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────

def init_state():
    ss = st.session_state
    if "initialized" in ss:
        return
    ss.initialized = True
    ss.work_root = tempfile.mkdtemp(prefix="boxing_app_")
    ss.stage = "upload"

    ss.match_name = ""
    ss.fps = 10
    ss.device = "cpu"

    ss.frame_names = []
    ss.n_frames = 0

    ss.max_stage_idx = 0

    # labelling
    ss.seed_indices = []
    ss.seed_ptr = 0
    ss.cur_seed_ptr_cached = None
    ss.cur_seed = None
    ss.assigned = {}
    ss.click_nonce = 0
    ss.confirmed_seeds = 0
    ss.seed_maps = []
    ss.first_confirmed_img_path = None
    ss.label_log = []
    ss.propagation_done = False

    # ring corners
    ss.corner_clicks = []
    ss.corner_nonce = 0
    ss.homography_accepted = False
    ss.homography_path = None  # set if user uploaded an existing homography.npy

    # training / weights
    ss.weights_path = None  # set if user uploaded existing weights, or after training
    ss.training_done = False

    # inference
    ss.inference_done = False
    ss.n_frames_processed = 0


def get_paths() -> pc.WorkPaths:
    return pc.WorkPaths(st.session_state.work_root)


def reset_all():
    ss = st.session_state
    try:
        shutil.rmtree(ss.work_root, ignore_errors=True)
    except Exception:
        pass
    for key in list(ss.keys()):
        del ss[key]
    init_state()


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def get_click(img_bgr: np.ndarray, key: str, display_width: int = 900):
    """Show img_bgr and return an (x, y) click in the IMAGE's native pixel
    coordinates (scaled back from whatever size it was displayed at), or
    None if no click has happened yet for this widget key."""
    h, w = img_bgr.shape[:2]
    disp_w = min(display_width, w)
    coords = streamlit_image_coordinates(to_rgb(img_bgr), width=disp_w, key=key)
    if not coords:
        return None
    disp_w_actual = coords.get("width") or disp_w
    disp_h_actual = coords.get("height") or (h * disp_w / w)
    scale_x = w / disp_w_actual
    scale_y = h / disp_h_actual
    return coords["x"] * scale_x, coords["y"] * scale_y


def sidebar_progress():
    ss = st.session_state
        
    cur_idx = STAGES.index(ss.stage)
    
    if cur_idx > ss.max_stage_idx:
        ss.max_stage_idx = cur_idx

    with st.sidebar:
        st.markdown("### Pipeline")
        
        for i, s in enumerate(STAGES):
            if i < cur_idx:
                label = f"✅ {STAGE_LABELS[s]}"
                is_disabled = False  # Can click to go backward
            elif i == cur_idx:
                label = f"➡️ {STAGE_LABELS[s]}"
                is_disabled = True   # Current stage, no need to click
            else:
                label = f"⬜ {STAGE_LABELS[s]}"
                # Disable the button if it's beyond the furthest stage reached
                is_disabled = (i > ss.max_stage_idx)
                
            if st.button(label, key=f"nav_{s}", use_container_width=True, disabled=is_disabled):
                ss.stage = s
                st.rerun()
                
        st.divider()
        if ss.match_name:
            st.caption(f"Match: **{ss.match_name}**")
            st.caption(f"Frames: {ss.n_frames}  ·  fps: {ss.fps}  ·  device: {ss.device}")
        st.divider()
        if st.button("🔁 Start over", use_container_width=True):
            reset_all()
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Upload
# ──────────────────────────────────────────────────────────────────────────────

def stage_upload():
    ss = st.session_state
    st.header("🥊 Boxing Position Tracker")
    st.write(
        "Upload a boxing match video. You'll then label the two boxers and the "
        "judge on a few frames, click the ring corners, and the pipeline will "
        "train a detector and track boxer foot positions through the whole clip."
    )

    uploaded = st.file_uploader("Match video", type=["mp4", "mov", "avi", "mkv", "m4v"])
    match_name = st.text_input(
        "Match name (used in output filenames)",
        value=ss.match_name,
        placeholder="e.g. Pergoliti_vs_Bogdanova_R1",
    )

    col1, col2 = st.columns(2)
    with col1:
        fps = st.slider(
            "Frame extraction rate (fps)", 1, 30, ss.fps,
            help="Lower fps = far fewer frames = much faster labelling, training, "
                 "and inference. Higher fps = smoother annotated output video.",
        )
    with col2:
        device = st.selectbox(
            "Compute device", ["cpu", "cuda"],
            index=["cpu", "cuda"].index(ss.device),
            help="Only choose 'cuda' if this machine has a working NVIDIA GPU + CUDA "
                 "PyTorch / onnxruntime-gpu install.",
        )

    with st.expander("Advanced — reuse existing files to skip steps"):
        st.caption(
            "Already trained a detector for these two boxers, or already have a "
            "homography for this camera angle? Upload them to skip straight past "
            "those steps."
        )
        weights_file = st.file_uploader("Existing trained weights (best.pt)", type=["pt"])
        homography_file = st.file_uploader("Existing homography (homography.npy)", type=["npy"])

    if not pc.check_ffmpeg():
        st.error("ffmpeg was not found on PATH. Install ffmpeg and restart the app before continuing.")

    start_disabled = uploaded is None or not pc.check_ffmpeg()
    if st.button("Start processing →", type="primary", disabled=start_disabled):
        paths = get_paths()
        with open(paths.uploaded_video, "wb") as f:
            f.write(uploaded.getbuffer())

        ss.match_name = match_name.strip() or os.path.splitext(uploaded.name)[0]
        ss.fps = fps
        ss.device = device

        if weights_file is not None:
            wpath = os.path.join(ss.work_root, "uploaded_best.pt")
            with open(wpath, "wb") as f:
                f.write(weights_file.getbuffer())
            ss.weights_path = wpath
            ss.training_done = True

        if homography_file is not None:
            hpath = os.path.join(ss.work_root, "uploaded_homography.npy")
            with open(hpath, "wb") as f:
                f.write(homography_file.getbuffer())
            ss.homography_path = hpath
            shutil.copy(hpath, paths.homography_npy)
            ss.homography_accepted = True

        with st.spinner("Extracting frames with ffmpeg…"):
            pc.extract_frames(paths.uploaded_video, paths.frames_dir, fps)
        ss.frame_names = pc.collect_frames(paths.frames_dir)
        ss.n_frames = len(ss.frame_names)
        ss.seed_indices = pc.sample_seed_indices(ss.n_frames, pc.NUM_SEED_FRAMES)

        if ss.weights_path and ss.homography_accepted:
            ss.stage = "infer"
        elif ss.weights_path:
            ss.stage = "corners"
        else:
            ss.stage = "label"
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Label boxers on seed frames (replaces cv2 label_tracks GUI)
# ──────────────────────────────────────────────────────────────────────────────

def stage_label():
    ss = st.session_state
    paths = get_paths()
    st.header("Step 2 — Label the boxers")

    # Advance ptr past any seed with fewer than 3 detected people, computing
    # (and caching) detection+tracking for the first workable seed.
    while ss.seed_ptr < len(ss.seed_indices):
        if ss.cur_seed_ptr_cached != ss.seed_ptr:
            seed_fi = ss.seed_indices[ss.seed_ptr]
            fname = ss.frame_names[seed_fi]
            img = cv2.imread(os.path.join(paths.frames_dir, fname))
            with st.spinner(f"Detecting people in seed frame {ss.seed_ptr + 1}/{len(ss.seed_indices)}…"):
                try:
                    track_ids, bboxes, tracker = pc.detect_and_track_seed(img, ss.device)
                except Exception as e:
                    st.error(f"Detection failed: {e}")
                    st.stop()
            if len(track_ids) < 3:
                ss.label_log.append(
                    f"Seed {ss.seed_ptr + 1} (frame {seed_fi}): fewer than 3 people "
                    "detected — skipped automatically."
                )
                ss.seed_ptr += 1
                continue
            ss.cur_seed_ptr_cached = ss.seed_ptr
            ss.cur_seed = {
                "fi": seed_fi, "fname": fname, "img": img,
                "track_ids": track_ids, "bboxes": bboxes, "tracker": tracker,
            }
            ss.assigned = {}
            ss.click_nonce += 1
        break

    if ss.seed_ptr >= len(ss.seed_indices):
        if ss.confirmed_seeds == 0:
            st.error(
                "Couldn't confirm any seed frames — fewer than 3 people were detected "
                "in every sampled frame. Try a video where the boxers and judge are "
                "clearly visible, or increase the extraction fps."
            )
            if st.button("⟲ Retry labelling"):
                ss.seed_ptr = 0
                ss.cur_seed_ptr_cached = None
                ss.confirmed_seeds = 0
                ss.seed_maps = []
                ss.label_log = []
                ss.propagation_done = False
                shutil.rmtree(paths.staging_dir, ignore_errors=True)
                os.makedirs(paths.staging_dir, exist_ok=True)
                st.rerun()
            return

        ss.stage = "corners"
        st.rerun()
        return

    cur = ss.cur_seed
    st.progress(ss.seed_ptr / len(ss.seed_indices))
    st.caption(f"Seed {ss.seed_ptr + 1} of {len(ss.seed_indices)} · video frame {cur['fi']} "
               f"· confirmed so far: {ss.confirmed_seeds}")

    n_assigned = len(ss.assigned)
    if n_assigned < 3:
        st.info(f"Click on the **{pc.IDENTITIES[n_assigned]}**.")
    else:
        st.success("All three identified — confirm below, or restart to re-click.")

    canvas = pc.render_track_assignment(cur["img"], cur["track_ids"], cur["bboxes"], ss.assigned)
    click_key = f"seed_click_{ss.seed_ptr}_{ss.click_nonce}_{n_assigned}"
    click = get_click(canvas, key=click_key)
    if click is not None and n_assigned < 3:
        x, y = click
        tid = pc.track_at(x, y, cur["track_ids"], cur["bboxes"])
        if tid in ss.assigned:
            st.warning("That person is already assigned — click a different person.")
        else:
            ss.assigned[tid] = pc.IDENTITIES[len(ss.assigned)]
            st.rerun()

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("↺ Restart clicks"):
            ss.assigned = {}
            ss.click_nonce += 1
            st.rerun()
    with b2:
        if st.button("Skip this seed"):
            ss.label_log.append(f"Seed {ss.seed_ptr + 1} (frame {cur['fi']}): skipped by user.")
            ss.seed_ptr += 1
            ss.cur_seed_ptr_cached = None
            st.rerun()
    with b3:
        if st.button("✓ Confirm & continue", type="primary", disabled=n_assigned < 3):
            identity_map = dict(ss.assigned)

            if ss.confirmed_seeds == 0:
                first_path = os.path.join(paths.staging_dir, "_first_confirmed.jpg")
                cv2.imwrite(first_path, cur["img"])
                ss.first_confirmed_img_path = first_path

            H_img, W_img = cur["img"].shape[:2]
            class_map = {ident: i for i, ident in enumerate(pc.IDENTITIES)}
            seed_lines = pc.build_label_lines(cur["track_ids"], cur["bboxes"], identity_map, class_map, W_img, H_img)
            pc.stage_annotated_frame(paths.staging_dir, cur["fi"], cur["fname"], cur["img"], seed_lines)

            ss.seed_maps.append({
		        "seed_frame": cur["fi"],
		        "track_to_identity": {str(k): v for k, v in identity_map.items()},
		        # Add these variables so we can use them in the bulk run at the end:
		        "identity_map": identity_map,
		        "tracker": cur["tracker"],
		        "W_img": W_img,
		        "H_img": H_img,
		    })
		
            ss.confirmed_seeds += 1
            ss.label_log.append(
                f"Seed {ss.seed_ptr + 1} (frame {cur['fi']}): confirmed — "
                + ", ".join(f"{v}→track {k}" for k, v in identity_map.items())
            )
            ss.seed_ptr += 1
            ss.cur_seed_ptr_cached = None
            st.rerun()

    if ss.confirmed_seeds > 0:
        st.divider()
        if st.button("Finish labelling now and continue →"):
            ss.seed_ptr = len(ss.seed_indices)
            st.rerun()

    if ss.label_log:
        with st.expander("Labelling log"):
            for line in ss.label_log:
                st.text(line)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Ring corners
# ──────────────────────────────────────────────────────────────────────────────

def stage_corners():
    ss = st.session_state
    paths = get_paths()
    st.header("Step 3 — Select the ring corners")

    img_path = ss.first_confirmed_img_path
    if not img_path:
        img_path = os.path.join(paths.frames_dir, ss.frame_names[0])
        st.caption("(Using the first video frame since labelling was skipped.)")
    img = cv2.imread(img_path)

    st.write("Click the 4 ring corners **in order**: top-left → top-right → bottom-right → bottom-left.")

    canvas = pc.render_corner_canvas(img, ss.corner_clicks)
    if len(ss.corner_clicks) < 4:
        next_label = pc.CORNER_LABELS[len(ss.corner_clicks)]
        st.info(f"Click: **{next_label}**  ({len(ss.corner_clicks)}/4)")
        click_key = f"corner_click_{ss.corner_nonce}_{len(ss.corner_clicks)}"
        click = get_click(canvas, key=click_key)
        if click is not None:
            ss.corner_clicks.append(click)
            st.rerun()
    else:
        st.image(to_rgb(canvas), width=900)

    if st.button("↺ Reset corners"):
        ss.corner_clicks = []
        ss.corner_nonce += 1
        ss.homography_accepted = False
        st.rerun()

    if len(ss.corner_clicks) == 4:
        H = pc.compute_homography(np.array(ss.corner_clicks, dtype=np.float32))
        warped = pc.render_warped_preview(img, H)
        st.subheader("Warped ring preview")
        st.image(to_rgb(warped), caption="Does the ring fill the frame correctly?", width=500)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✓ Accept homography", type="primary"):
                np.save(paths.homography_npy, H)
                ss.homography_accepted = True
                ss.stage = "train"
                st.rerun()
        with c2:
            if st.button("Retry corner clicks"):
                ss.corner_clicks = []
                ss.corner_nonce += 1
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 — Train YOLO
# ──────────────────────────────────────────────────────────────────────────────

def stage_train():
    ss = st.session_state
    paths = get_paths()
    st.header("Step 4 — Train the boxer detector")

    if ss.weights_path or ss.training_done:
        ss.stage = "infer"
        st.rerun()
        return

    st.write(f"Confirmed seed labels staged for training: **{ss.confirmed_seeds}**")

    c1, c2 = st.columns(2)
    epochs = c1.number_input("Training epochs", min_value=5, max_value=300, value=60, step=5,
                             help="How many times the model studies the labeled frames. "
                                  "Higher = better tracking and accuracy but takes longer to train and may result in overfitting. "
                                  "Lower = faster training, but the model might lose track of the boxers easily. ")
    batch = c2.number_input(
        "Batch size", min_value=2, max_value=64, value=16, step=2,
        help="How many images the model processes at once. "
             "Higher = faster and smoother training, but requires a lot of GPU memory (can cause crashes). "
             "Lower = safer for less powerful hardware but slightly slower."
    )

    if not ss.training_done:
        if st.button("Start training", type="primary"):

            # bulk propagation block
            if not ss.propagation_done:
                with st.spinner(f"Propagating labels for all {ss.confirmed_seeds} confirmed seeds…"):
                    class_map = {ident: i for i, ident in enumerate(pc.IDENTITIES)}
                    for seed_data in ss.seed_maps:
                        window = pc.propagate_window(
                            seed_data["tracker"], paths.frames_dir, ss.frame_names, 
                            seed_data["seed_frame"], ss.n_frames, ss.device
                        )
                        for fi, fname, img, tids, bxs in window:
                            lines = pc.build_label_lines(
                                tids, bxs, seed_data["identity_map"], class_map, 
                                seed_data["W_img"], seed_data["H_img"]
                            )
                            pc.stage_annotated_frame(paths.staging_dir, fi, fname, img, lines)
                ss.propagation_done = True

            try:
                with st.spinner("Building YOLO dataset from staged frames…"):
                    stats = pc.finalize_dataset(paths.staging_dir, paths.dataset_dir, paths.dataset_yaml, pc.CLASS_NAMES)
                st.write(f"Dataset ready — {stats['train']} train / {stats['val']} val images.")
            except Exception as e:
                st.error(f"Could not build the dataset: {e}")
                st.stop()

            progress_bar = st.progress(0.0, text="Training…")

            def on_epoch(epoch, total):
                progress_bar.progress(min(epoch / total, 1.0), text=f"Epoch {epoch}/{total}")

            try:
                weights_path = pc.train_yolo(
                    paths.dataset_yaml, int(epochs), int(batch), ss.device, paths.runs_dir,
                    progress_cb=on_epoch
                )
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()

            # go straight to inference
            ss.weights_path = weights_path
            ss.training_done = True
            ss.stage = "infer"
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Stage 5 — Inference
# ──────────────────────────────────────────────────────────────────────────────

def stage_infer():
    ss = st.session_state
    paths = get_paths()
    st.header("Step 5 — Run detection + pose inference")

    if not ss.inference_done:
        if st.button("▶ Run inference", type="primary"):
            try:
                with st.spinner("Loading YOLO + RTMPose models…"):
                    yolo_model = pc.load_yolo_model(ss.weights_path)
                    pose_model = pc.load_pose_model(ss.device)
            except Exception as e:
                st.error(f"Could not load models: {e}")
                st.stop()

            n = int(ss.n_frames)

            progress = st.progress(0.0)
            status = st.empty()
            preview = st.empty()
            all_results = {}

            for i, fname in enumerate(ss.frame_names[:n]):
                img = cv2.imread(os.path.join(paths.frames_dir, fname))
                if img is None:
                    continue
                annotated, record = pc.run_inference_frame(img, i, yolo_model, pose_model)
                cv2.imwrite(os.path.join(paths.output_frames, fname), annotated)
                all_results[i] = {"frame": fname, "people": record}

                if (i + 1) % 5 == 0 or i == n - 1:
                    progress.progress((i + 1) / n)
                    status.text(f"{i + 1}/{n} frames processed")
                if (i + 1) % 30 == 0 or i == n - 1:
                    preview.image(to_rgb(annotated), caption=f"frame {i}", width=500)

            header, rows = pc.rows_from_results(all_results, 1, ss.match_name)
            pc.write_csv(paths.ankle_csv, header, rows)

            ss.inference_done = True
            ss.n_frames_processed = n
            st.success(f"Inference complete on {n} frames.")
            st.rerun()
    else:
        st.success(f"Inference complete on {ss.n_frames_processed} frames.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("↺ Re-run inference"):
                ss.inference_done = False
                st.rerun()
        with c2:
            if st.button("View results →", type="primary"):
                ss.stage = "results"
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 6 — Results
# ──────────────────────────────────────────────────────────────────────────────

def stage_results():
    ss = st.session_state
    paths = get_paths()
    st.header("Step 6 — Validate & download")

    hpath = paths.homography_npy if os.path.exists(paths.homography_npy) else ss.homography_path
    H = np.load(hpath)

    if not os.path.exists(paths.output_video):
        with st.spinner("Stitching annotated frames into a video…"):
            try:
                pc.stitch_video(paths.output_frames, paths.output_video, ss.fps)
            except Exception as e:
                st.error(f"Could not stitch video: {e}")
                st.stop()

    st.subheader("🎬 Annotated match video")
    st.caption("Use this to visually confirm the pose detections and boxer identities look correct.")
    st.video(paths.output_video)
    with open(paths.output_video, "rb") as f:
        st.download_button(
            "⬇ Download annotated video (mp4)", f,
            file_name=f"{ss.match_name}_annotated.mp4", mime="video/mp4",
        )

    st.subheader("📄 Ankle position data")
    df = pd.read_csv(paths.ankle_csv)
    enriched = pc.warp_and_enrich(df, H)
    enriched.to_csv(paths.enriched_csv, index=False)

    m1, m2, m3 = st.columns(3)
    m1.metric("Frames processed", df["frame_idx"].nunique())
    for col, ident in zip((m2, m3), ("Blue_boxer", "Red_boxer")):
        sub = df[df["identity"] == ident]
        rate = (sub["status"] == "detected").mean() * 100 if len(sub) else 0
        col.metric(f"{ident.replace('_', ' ')} detection rate", f"{rate:.0f}%")

    st.caption("**Note:** For frames where a boxer is temporarily obscured or undetected, their position data is automatically interpolated.")

    st.dataframe(enriched.head(300), use_container_width=True)

    st.download_button(
        "⬇ Download ring-warped ankle position csv file", enriched.to_csv(index=False),
        file_name=f"boxer_positions_{ss.match_name}.csv", mime="text/csv",
    )

    st.divider()
    if st.button("🔁 Process another video"):
        reset_all()
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    init_state()
    sidebar_progress()
    stage_fn = {
        "upload": stage_upload,
        "label": stage_label,
        "corners": stage_corners,
        "train": stage_train,
        "infer": stage_infer,
        "results": stage_results,
    }[st.session_state.stage]
    stage_fn()


if __name__ == "__main__":
    main()
