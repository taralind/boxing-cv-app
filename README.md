# Boxing Position Tracker

Streamlit web app that turns a boxing match video into a csv of each boxer's feet positions on the ring, frame-by-frame over the match. The pipeline does boxer tracking, pose estimation, and ring-geometry warping, with only some slight manual input required. Position data can be downloaded and used for position, ring control and ring dominance analysis. 

## Detection preview

https://github.com/user-attachments/assets/c766bf49-81ad-400b-8a9f-c7233568ca2c

## App Demo

https://github.com/user-attachments/assets/e43d2967-194a-4b69-b8e5-2774d8d7ea38

## How it works

The app takes the user through six guided stages:

| Stage | What happens |
|---|---|
| **1. Upload** | Upload a match video (single static camera, whole ring in frame, trimmed to just the round). Set the frame extraction rate (fps) and compute device (CPU/GPU). Optionally upload an already-trained detector (`best.pt`) or an existing ring homography (`homography.npy`) to skip straight past those steps. |
| **2. Label boxers** | The app samples ~10 seed frames spread across the video, auto-detects people in each using a general-purpose pose model + tracker, and asks the user to click on the **Blue boxer**, **Red boxer**, and **Judge** in turn. Confirmed labels are propagated across a short window of surrounding frames. |
| **3. Ring corners** | Click the four ring corners (top-left → top-right → bottom-right → bottom-left) on a reference frame. This computes a homography used later to warp raw pixel coordinates into real ring-space coordinates. |
| **4. Train model** | A custom YOLOv8 detector is fine-tuned on the labelled frames to recognise the Blue boxer, Red boxer, and Judge specifically in this match (epochs/batch size configurable). |
| **5. Run inference** | The trained detector + a pose estimation model (RTMPose) run over every extracted frame, drawing skeleton/ankle overlays and recording keypoints. |
| **6. Results** | Annotated frames are stitched into a downloadable video. Ankle positions are interpolated across any frames where a boxer was briefly lost, then warped into ring coordinates using the homography from Step 3. Both the video and the CSV are available to download. |

## Outputs

- **Annotated video** (`.mp4`) — the original footage with pose skeletons, ankle markers, and identity labels overlaid, for visually validating that tracking worked.
- **Ring-warped position CSV** (`.csv`) — per-frame ankle coordinates for each boxer, translated into ring-relative coordinates.

## Data dictionary

Starting point based on the current CSV output:

| Column | Description |
|---|---|
| `Round` | Round number (currently always `1`; see Roadmap re: multi-round support). |
| `match_name` | Name given to the match at upload time; used to identify the source video. |
| `frame_idx` | Index of the frame within the extracted sequence. |
| `frame_file` | Filename of the source frame image. |
| `identity` | Which tracked subject the row belongs to: `Blue_boxer`, `Red_boxer`, or `Judge`. |
| `status` | Detection status for that identity in that frame: `detected`, `box_only` (detected but no pose), `lost`, or `interpolated` (position filled in from neighbouring frames). |
| `left_ankle_x` / `left_ankle_y` / `left_ankle_conf` | Raw pixel coordinates and confidence for the left ankle keypoint. |
| `right_ankle_x` / `right_ankle_y` / `right_ankle_conf` | Raw pixel coordinates and confidence for the right ankle keypoint. |
| `left_ankle_warped_x` / `left_ankle_warped_y` | Left ankle position warped into ring-space coordinates. |
| `right_ankle_warped_x` / `right_ankle_warped_y` | Right ankle position warped into ring-space coordinates. |
| `left_ankle_rel_x` / `left_ankle_rel_y` | Left ankle position relative to the centre of the ring. |
| `right_ankle_rel_x` / `right_ankle_rel_y` | Right ankle position relative to the centre of the ring. |
| `left_ankle_dist_cm` / `right_ankle_dist_cm` | Straight-line distance of each ankle from the centre of the ring, in centimetres. |

## Running the app

```bash
pip install streamlit streamlit-image-coordinates opencv-python numpy pandas ultralytics rtmlib supervision ffmpeg
streamlit run app.py
```

Then open the local URL Streamlit prints in your browser and follow the six-step workflow above.

## Repository contents

- `app.py` — the Streamlit interface: session state, page layout, and the six pipeline stages.
- `pipeline_core.py` — the underlying computer vision pipeline: frame extraction, detection/tracking, labelling helpers, YOLO training, inference, and ring-warping logic. Can be reused outside Streamlit if needed.

## Planned additions

- **Combine round videos** — support stitching/analysing multiple rounds of the same match together rather than one video at a time.
- **Video size & robustness testing** — explore practical limits on video length/resolution, and test performance across a wider range of real competition footage (different camera angles, lighting, ring setups).
- **Deeper results analysis** — expand the results section with more built-in analysis, potentially an insights dashboard, rather than just raw video + CSV output.
