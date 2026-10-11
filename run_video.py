"""
run_video.py  —  Crowd density estimation on multi-camera video feeds.

OUTPUT (video_outputs/)
  result_<NNNNNN>.png   3-panel figure every SAVE_EVERY_N frames
                        (camera overlay | density gradient | voxel BEV)
  results.csv           One row per processed frame, overwritten fresh each run

USAGE
  .\.crowd\Scripts\python.exe run_video.py
"""

from __future__ import annotations
import csv, os, sys, time, traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crowd3d.geometry import Camera, VoxelGrid
from crowd3d.config import SceneBounds, ReconstructionConfig, CountingConfig
from crowd3d.detection import DetectedBlob, ViewDetections
from crowd3d.reconstruction import (
    carve_voxels, ray_vote_occupancy, project_to_ground,
    resample_to_citystreet_groundplane, ReconstructionResult,
)
from crowd3d.counting import estimate_crowd

# ── OUTPUT ────────────────────────────────────────────────────────────────────
OUT_DIR = "video_outputs"

# ══════════════════════════════════════════════════════════════════════════════
#   CAMERA VIDEO SOURCES  —  edit here
#   source: file path | folder of frame_XXXX.jpg images | "rtsp://..." | int (webcam)
#   calib:  "citystreet_view1/2/3"  |  {}  (auto-pinhole)  |  custom dict
# ══════════════════════════════════════════════════════════════════════════════
CAMERA_VIDEO_SOURCES = [
    {
        "source":  r"CityStreet\CityStreet_zip\CityStreet\CityStreet\image_frames\camera1",
        "view_id": "cam_A",
        "calib":   "citystreet_view1",
    },
    {
        "source":  r"CityStreet\CityStreet_zip\CityStreet\CityStreet\image_frames\camera3",
        "view_id": "cam_B",
        "calib":   "citystreet_view2",
    },
    {
        "source":  r"CityStreet\CityStreet_zip\CityStreet\CityStreet\image_frames\camera4",
        "view_id": "cam_C",
        "calib":   "citystreet_view3",
    },
    # add more entries freely — any number works
    # {"source": "rtsp://...", "view_id": "cam_D", "calib": {}},
    # {"source": 0,            "view_id": "cam_E", "calib": {}},  # webcam
]

# ── PIPELINE SETTINGS ─────────────────────────────────────────────────────────
BG_BUILD_FRAMES  = 20       # frames per camera used to build initial background
BG_UPDATE_ALPHA  = 0.02     # background EMA update (0 = static background)
MAX_FRAMES       = 500      # max frames to process (0 = unlimited)
SAVE_EVERY_N     = 10       # save result image every N processed frames
FOLDER_START     = 636      # first frame index for folder-mode sources
FOLDER_END       = 1634
FOLDER_STEP      = 2

# ── SCENE (metres) ────────────────────────────────────────────────────────────
SCENE = dict(x_min=-18., x_max=25., y_min=-28., y_max=24.,
             z_min=0.,   z_max=2.2, cell_xy=0.35, cell_z=0.30)

# ── DETECTION ─────────────────────────────────────────────────────────────────
DIFF_THRESHOLD = 24
MIN_BLOB_AREA  = 50

# ── COUNTING ──────────────────────────────────────────────────────────────────
COUNTING = dict(gaussian_sigma_m=0.40, peak_min_distance_m=0.65,
                peak_rel_threshold=0.20, peak_abs_threshold=0.05,
                person_footprint_m2=1.05)


# ── KNOWN CITYSTREET CALIBRATIONS ─────────────────────────────────────────────
_CS = {
    "citystreet_view1": dict(
        img_w=2704, img_h=1520, cx=1384.55, cy=747.8465,
        dpx=1./1243.4, dpy=1./1248.9, focal=1., sx=1., kappa1=-0.2090,
        R=np.array([[-0.03615359, 0.99596323,-0.0821594 ],
                    [-0.34251474, 0.06488427, 0.93726927],
                    [ 0.93881658, 0.06202646, 0.33878628]], np.float64),
        t=np.array([1275.94012233, 226.52473842, 24230.24027107], np.float64)),
    "citystreet_view2": dict(
        img_w=2704, img_h=1520, cx=1292.5376, cy=770.5181,
        dpx=1./1258.18922, dpy=1./1259.88567, focal=1., sx=1., kappa1=-0.23977946,
        R=np.array([[-0.99730688, 0.06324747, 0.03713147],
                    [ 0.02439897,-0.19132768, 0.9812229 ],
                    [ 0.06916415, 0.97948633, 0.18926923]], np.float64),
        t=np.array([9680.47673912, 2273.39720387, 31436.76306171], np.float64)),
    "citystreet_view3": dict(
        img_w=2704, img_h=1520, cx=1374.1, cy=785.7148,
        dpx=1./1255.9, dpy=1./1257.2, focal=1., sx=1., kappa1=-0.2489,
        R=np.array([[ 0.99545502, 0.07578782,-0.05766726],
                    [ 0.0167508,  0.45675628, 0.88943415],
                    [ 0.09374816,-0.88635766, 0.45341083]], np.float64),
        t=np.array([3309.55016035,-4603.90350916, 29264.19379322], np.float64)),
}


def _build_camera(view_id, calib, img_w, img_h) -> Camera:
    if isinstance(calib, str) and calib in _CS:
        return Camera(view_id=view_id, camera_name=calib, **_CS[calib])
    if isinstance(calib, dict) and calib:
        return Camera(view_id=view_id,
                      camera_name=calib.get("camera_name", view_id),
                      img_w=calib.get("img_w",img_w), img_h=calib.get("img_h",img_h),
                      cx=calib.get("cx",img_w/2.), cy=calib.get("cy",img_h/2.),
                      dpx=calib.get("dpx",1./(img_w*.5)),
                      dpy=calib.get("dpy",1./(img_h*.5)),
                      focal=calib.get("focal",1.), sx=calib.get("sx",1.),
                      kappa1=calib.get("kappa1",0.),
                      R=np.array(calib["R"],np.float64),
                      t=np.array(calib["t"],np.float64))
    fov = 70.
    f   = (img_w/2.) / np.tan(np.radians(fov/2.))
    cx_s = (SCENE["x_min"]+SCENE["x_max"])/2.
    cy_s = (SCENE["y_min"]+SCENE["y_max"])/2.
    h_m  = 8.
    yaw  = np.radians(abs(hash(view_id)) % 360)
    cx_c = cx_s + 5.*np.cos(yaw); cy_c = cy_s + 5.*np.sin(yaw)
    fwd  = np.array([cx_s-cx_c, cy_s-cy_c, -h_m], np.float64); fwd /= np.linalg.norm(fwd)
    up   = np.array([0.,-1.,0.], np.float64)
    r    = np.cross(fwd, up); r /= np.linalg.norm(r)
    up   = np.cross(r, fwd)
    R    = np.stack([r, -up, fwd])
    t    = -R @ np.array([cx_c*1000., cy_c*1000., -h_m*1000.], np.float64)
    return Camera(view_id=view_id, camera_name=f"auto_{view_id}",
                  img_w=img_w, img_h=img_h, cx=img_w/2., cy=img_h/2.,
                  dpx=1./f, dpy=1./f, focal=1., sx=1., kappa1=0., R=R, t=t)


# ── FOLDER-MODE READER ────────────────────────────────────────────────────────
class FolderCapture:
    def __init__(self, folder):
        self.folder  = folder
        self.indices = list(range(FOLDER_START, FOLDER_END+1, FOLDER_STEP))
        self._pos    = 0

    def isOpened(self):  return self._pos < len(self.indices)
    def release(self):   pass
    def get(self, _):    return 0.

    def read(self):
        while self._pos < len(self.indices):
            idx  = self.indices[self._pos]; self._pos += 1
            path = os.path.join(self.folder, f"frame_{idx:04d}.jpg")
            if not os.path.exists(path): continue
            img = cv2.imread(path)
            if img is not None: return True, img
        return False, None


def _open(source) -> Optional[Any]:
    if isinstance(source, int):
        cap = cv2.VideoCapture(source); return cap if cap.isOpened() else None
    if isinstance(source, str):
        if os.path.isdir(source):
            fc = FolderCapture(source); return fc if fc.isOpened() else None
        cap = cv2.VideoCapture(source); return cap if cap.isOpened() else None
    return None


# ── PER-CAMERA STATE ──────────────────────────────────────────────────────────
@dataclass
class CamState:
    view_id: str
    camera:  Camera
    source:  Any
    active:  bool = True
    fails:   int  = 0
    MAX_FAIL: int = 5

    bg_frames: List[np.ndarray] = field(default_factory=list)
    bg:        Optional[np.ndarray] = None
    bg_run:    Optional[np.ndarray] = None
    n_read:    int = 0

    def read(self):
        try:
            ok, f = self.source.read()
            if not ok or f is None:
                self.fails += 1
                if self.fails >= self.MAX_FAIL: self.active = False
                return None
            self.fails  = 0
            self.n_read += 1
            return f
        except Exception:
            self.fails += 1
            if self.fails >= self.MAX_FAIL: self.active = False
            return None

    def update_bg(self, frame):
        if len(self.bg_frames) < BG_BUILD_FRAMES:
            self.bg_frames.append(frame.copy())
            if len(self.bg_frames) == BG_BUILD_FRAMES:
                stack    = np.stack(self.bg_frames, 0).astype(np.float32)
                self.bg  = np.median(stack, 0).astype(np.uint8)
                self.bg_run = self.bg.astype(np.float32)
                print(f"  [BG] {self.view_id} ready ({BG_BUILD_FRAMES} frames)")
        elif self.bg_run is not None and BG_UPDATE_ALPHA > 0:
            self.bg_run = (1.-BG_UPDATE_ALPHA)*self.bg_run + BG_UPDATE_ALPHA*frame.astype(np.float32)
            self.bg     = self.bg_run.astype(np.uint8)

    @property
    def bg_ready(self): return self.bg is not None


# ── DETECTION ─────────────────────────────────────────────────────────────────
def _detect(img_bgr, bg_bgr, view_id) -> ViewDetections:
    diff = cv2.absdiff(img_bgr, bg_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    kc = cv2.getStructuringElement(cv2.MORPH_RECT,    (5,10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ko)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs, overlay = [], img_bgr.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_BLOB_AREA: continue
        x,y,bw,bh = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx_b = M["m10"]/M["m00"]; cy_b = M["m01"]/M["m00"]
        blobs.append(DetectedBlob(view_id=view_id,
                                  centroid_px=(cx_b,cy_b),
                                  head_px=(cx_b, y+.15*bh),
                                  bbox_px=(x,y,bw,bh), area_px=area, contour=c))
        cv2.rectangle(overlay,(x,y),(x+bw,y+bh),(0,220,0),2)
        cv2.circle(overlay,(int(cx_b),int(y+.15*bh)),5,(0,0,255),-1)
    return ViewDetections(view_id=view_id, foreground_mask=mask,
                          downscaled_mask=mask, blobs=blobs, overlay_img=overlay)


# ── OUTPUT ────────────────────────────────────────────────────────────────────
def _tile(overlays: dict, target_h=480) -> np.ndarray:
    imgs = []
    for img in overlays.values():
        h,w = img.shape[:2]
        s = target_h/h
        imgs.append(cv2.resize(img, (int(w*s), target_h)))
    if len(imgs) == 1: return imgs[0]
    rows = []
    for i in range(0, len(imgs), 2):
        pair = imgs[i:i+2]
        if len(pair) == 1: pair.append(np.zeros_like(pair[0]))
        rows.append(np.hstack(pair))
    max_w = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        if r.shape[1] < max_w:
            r = np.hstack([r, np.zeros((r.shape[0],max_w-r.shape[1],3),np.uint8)])
        out.append(r)
    return np.vstack(out)


def _save_result(frame_idx, n_active, detections, grid, recon, counting, pred_pts_m, path):
    overlays = {vid: d.overlay_img for vid,d in detections.items() if d.overlay_img is not None}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#333333")

    # Panel 1 — camera(s) with detected persons
    if overlays:
        axes[0].imshow(cv2.cvtColor(_tile(overlays), cv2.COLOR_BGR2RGB))
    axes[0].set_title(
        f"Detected Persons  |  Frame {frame_idx}  |  {n_active} cam(s)  |  "
        f"{sum(len(d.blobs) for d in detections.values())} blobs",
        color="white", fontsize=10, pad=6)
    axes[0].axis("off")

    # Panel 2 — density gradient
    ext = grid.bev_extent()
    im2 = axes[1].imshow(counting.density_map_metric,
                          extent=ext, origin="lower", cmap="inferno", aspect="auto")
    if len(pred_pts_m) > 0:
        axes[1].scatter(pred_pts_m[:,0], pred_pts_m[:,1], c="white", s=18,
                        marker="o", edgecolors="#00ccff", linewidths=0.9, zorder=5,
                        label=f"{counting.peak_count} persons")
        axes[1].legend(loc="upper right", facecolor="#222222",
                       edgecolor="none", labelcolor="white", fontsize=8)
    axes[1].set_title(
        f"Crowd Density  |  Peak: {counting.peak_count}  Mass: {counting.mass_count:.1f}",
        color="white", fontsize=10, pad=6)
    axes[1].set_xlabel("X (m)", color="#aaaaaa", fontsize=8)
    axes[1].set_ylabel("Y (m)", color="#aaaaaa", fontsize=8)
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 — voxel BEV
    im3 = axes[2].imshow(recon.ground_occupancy_metric,
                          extent=ext, origin="lower", cmap="magma", aspect="auto")
    axes[2].set_title(
        f"Voxel Occupancy  |  {int(recon.occupancy_vol.sum())} voxels",
        color="white", fontsize=10, pad=6)
    axes[2].set_xlabel("X (m)", color="#aaaaaa", fontsize=8)
    axes[2].set_ylabel("Y (m)", color="#aaaaaa", fontsize=8)
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout(pad=1.5)
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    plt.close(fig)


# ── CSV ───────────────────────────────────────────────────────────────────────
CSV_FIELDS = ["frame", "timestamp", "active_cams", "peak_count",
              "mass_count", "occupied_voxels", "total_blobs", "failed_cams"]

def _init_csv():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    return path

def _append_csv(csv_path, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Crowd3D | video mode")

    # 1. Open sources
    states: Dict[str, CamState] = {}
    for entry in CAMERA_VIDEO_SOURCES:
        vid, src, calib = entry["view_id"], entry["source"], entry.get("calib",{})
        cap = _open(src)
        if cap is None:
            print(f"  SKIP {vid}: cannot open '{src}'"); continue
        ok, probe = cap.read()
        if not ok or probe is None:
            print(f"  SKIP {vid}: first frame failed"); cap.release(); continue
        h,w = probe.shape[:2]
        cam = _build_camera(vid, calib, w, h)
        if isinstance(cap, FolderCapture): cap._pos = 0
        else: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        states[vid] = CamState(view_id=vid, camera=cam, source=cap)
        print(f"  {vid}: {w}x{h}  source='{src}'")

    if not states:
        sys.exit("No valid video sources.")

    # 2. Scene grid
    grid    = VoxelGrid(SceneBounds(**SCENE))
    cnt_cfg = CountingConfig(**COUNTING)
    print(f"  Voxel grid: {grid.shape} = {grid.n_voxels} voxels")

    # 3. CSV — overwrite fresh for this run
    csv_path = _init_csv()

    # 4. Main loop
    print("  Processing...  (Ctrl+C to stop)\n")
    frame_idx = 0; processed = 0; counts = []
    t0 = time.time()

    try:
        while True:
            active = [v for v,s in states.items() if s.active]
            if not active: print("All feeds ended."); break
            if MAX_FRAMES > 0 and processed >= MAX_FRAMES:
                print(f"Reached MAX_FRAMES={MAX_FRAMES}."); break

            frame_idx += 1

            # read frames
            frames: Dict[str, np.ndarray] = {}
            for vid in list(active):
                st = states[vid]
                f  = st.read()
                if f is None:
                    if not st.active:
                        print(f"  [LOST] {vid}")
                else:
                    frames[vid] = f
                    st.update_bg(f)

            if not frames: continue
            processed += 1

            # detect (only cams with BG ready)
            detections: Dict[str, ViewDetections] = {}
            cameras_rdy: Dict[str, Camera]        = {}
            for vid, fr in frames.items():
                st = states[vid]
                if not st.bg_ready: continue
                detections[vid]   = _detect(fr, st.bg, vid)
                cameras_rdy[vid]  = st.camera

            warming = sum(1 for s in states.values() if s.active and not s.bg_ready)
            if not cameras_rdy:
                print(f"  frame {frame_idx:6d} | warming {warming} cam(s)...", end="\r")
                continue

            # reconstruct
            n = len(cameras_rdy)
            rec_cfg = ReconstructionConfig(
                min_camera_consensus=max(1,min(2,n)),
                consensus_ratio=0.5 if n>=2 else 0.3,
                ray_voting_sigma_px=20.)
            try:
                occ,conf,cons = carve_voxels(grid, cameras_rdy, detections, rec_cfg)
                vote = ray_vote_occupancy(grid, cameras_rdy, detections, sigma_px=20.)
                v_n  = vote / max(float(vote.max()), 1e-6)
                bev  = project_to_ground(conf*0.4+v_n*0.6, grid, z_min_m=0.5, z_max_m=2.0)
                recon = ReconstructionResult(occupancy_vol=occ, confidence_vol=conf,
                                             consensus_count=cons,
                                             ground_occupancy_metric=bev,
                                             ground_occupancy_citystreet=None)
                result = estimate_crowd(bev, grid, cnt_cfg)
            except Exception as e:
                print(f"\n  [WARN] frame {frame_idx}: {e}"); continue

            pred = (np.array([[p.x_m,p.y_m,p.z_m] for p in result.detected_persons],np.float64)
                    if result.detected_persons else np.zeros((0,3)))

            counts.append(result.peak_count)
            failed = [v for v,s in states.items() if not s.active]

            # console
            blobs_str = "  ".join(f"{v}:{len(detections[v].blobs)}" for v in cameras_rdy)
            fail_str  = f"  FAILED:{failed}" if failed else ""
            print(f"  frame {frame_idx:6d} | {n} cam(s) | blobs[{blobs_str}] | "
                  f"peak:{result.peak_count:3d}  mass:{result.mass_count:5.1f}{fail_str}")

            # CSV row
            _append_csv(csv_path, {
                "frame":          frame_idx,
                "timestamp":      datetime.now().strftime("%H:%M:%S"),
                "active_cams":    n,
                "peak_count":     result.peak_count,
                "mass_count":     round(result.mass_count,1),
                "occupied_voxels":int(occ.sum()),
                "total_blobs":    sum(len(d.blobs) for d in detections.values()),
                "failed_cams":    len(failed),
            })

            # save result image
            if processed % SAVE_EVERY_N == 0:
                img_path = os.path.join(OUT_DIR, f"result_{frame_idx:06d}.png")
                try:
                    _save_result(frame_idx, n, detections, grid, recon, result, pred, img_path)
                    print(f"  -> {img_path}")
                except Exception as e:
                    print(f"  [WARN] save failed: {e}")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    # 5. Summary
    for s in states.values(): s.source.release()
    elapsed = time.time() - t0
    print(f"\n  {processed} frames in {elapsed:.1f}s  ({processed/max(elapsed,1):.1f} fps)")
    if counts:
        print(f"  Count — mean:{np.mean(counts):.1f}  "
              f"min:{min(counts)}  max:{max(counts)}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
