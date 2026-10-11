"""run_images.py: YOLO person detection + ground-plane crowd map on still images.

Output (image_outputs/):
    result.png    cameras with boxes | ground-plane detections | crowd density
    results.csv   one row, regenerated each run

Usage:
    .\\.crowd\\Scripts\\python.exe run_images.py
"""
from __future__ import annotations

import csv
import os
import re
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crowd3d.crowd3d_all import (
    GroundGrid, PersonDetector, YoloConfig, boxes_to_ground, build_camera, draw_view, eval_view,
    fuse_views, gt_points_for_frame, ground_density, load_gt_groundplane, load_roi_mask, load_via_points,
    match_points, plot_result, prf,
)

OUT_DIR = "image_outputs"

# --- CAMERA SOURCES: edit paths. calib: "citystreet_view1/2/3" | dict(R, t, cx, cy, dpx, dpy, kappa1, ...) | {} ---
CAMERA_SOURCES = [
    {"image_path": r"CityStreet\CityStreet_zip\CityStreet\CityStreet\image_frames\camera1\frame_0636.jpg",
     "view_id": "cam_A", "calib": "citystreet_view1"},
    {"image_path": r"CityStreet\CityStreet_zip\CityStreet\CityStreet\image_frames\camera3\frame_0636.jpg",
     "view_id": "cam_B", "calib": "citystreet_view2"},
    {"image_path": r"CityStreet\CityStreet_zip\CityStreet\CityStreet\image_frames\camera4\frame_0636.jpg",
     "view_id": "cam_C", "calib": "citystreet_view3"},
]

# --- YOLO ---
YOLO = dict(model="yolo11x.pt", device=None, conf=0.20, imgsz_full=1280,
            use_full=True, use_tiles=True, tile=1024, tile_overlap=0.25, tile_imgsz=1024)

# --- GROUND PLANE ---
GROUND = dict(mode="head", head_height_m=1.70, fuse_radius_m=1.0, min_views=1, density_sigma_m=0.6)
SCENE = dict(x_min=-18.0, x_max=25.0, y_min=-28.0, y_max=24.0, cell=0.25)

# --- OPTIONAL: camera-view ROI filter and ground-truth evaluation (set to None to skip) ---
CAMERA_ROI_DIR = r"CityStreet\CityStreet_zip\CityStreet\CityStreet\ROI_maps\ROIs\camera_view"
LABELS = dict(gt_h5=r"CityStreet\CityStreet_correctedlabels\Street_groundplane_pmap.h5",
              via_dir=r"CityStreet\CityStreet_correctedlabels", match_radius_m=1.0)


def _p(path):
    return os.path.normpath(path.replace("\\", os.sep)) if path else path


def _view_index(calib):
    m = re.fullmatch(r"citystreet_view(\d)", calib) if isinstance(calib, str) else None
    return int(m.group(1)) if m else None


def _frame_id(path):
    m = re.search(r"frame_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _write_csv(row):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    return path


def main():
    t0 = time.time()
    print("Crowd3D | image mode | YOLO")

    views = []
    for e in CAMERA_SOURCES:
        path = _p(e["image_path"])
        if not os.path.exists(path):
            print(f"  SKIP {e['view_id']}: not found: {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"  SKIP {e['view_id']}: imread failed")
            continue
        h, w = img.shape[:2]
        calib = e.get("calib", {})
        idx = _view_index(calib)
        roi = load_roi_mask(_p(CAMERA_ROI_DIR), idx) if (CAMERA_ROI_DIR and idx) else None
        views.append(dict(id=e["view_id"], path=path, img=img, cam=build_camera(e["view_id"], calib, w, h),
                          idx=idx, roi=roi, frame=_frame_id(path)))
        print(f"  {e['view_id']}: {w}x{h} | calibrated={views[-1]['cam'] is not None} | roi={roi is not None}")
    if not views:
        sys.exit("No valid images loaded.")

    detector = PersonDetector(YoloConfig(**YOLO))
    for v in views:
        t1 = time.time()
        v["boxes"], v["scores"] = detector.detect(v["img"], v["roi"])
        print(f"  {v['id']}: {len(v['boxes'])} persons ({time.time() - t1:.1f}s)")

    grid = GroundGrid(**SCENE)
    per_view = {}
    for v in views:
        if v["cam"] is None:
            continue
        xy, ok = boxes_to_ground(v["cam"], v["boxes"], GROUND["mode"], GROUND["head_height_m"],
                                 (grid.x_min - 10, grid.x_max + 10, grid.y_min - 10, grid.y_max + 10))
        per_view[v["id"]] = (xy[ok], v["scores"][ok])
    fused = np.zeros((0, 2))
    nviews = np.zeros(0, int)
    if per_view:
        fused, nviews, _ = fuse_views(per_view, GROUND["fuse_radius_m"])
        keep = nviews >= min(GROUND["min_views"], len(per_view))
        fused, nviews = fused[keep], nviews[keep]
        count = len(fused)
    else:
        count = max(len(v["boxes"]) for v in views)
    density = ground_density(fused, grid, GROUND["density_sigma_m"]) if per_view else None
    print(f"  persons: {count}" + (f" (fused from {len(per_view)} calibrated view(s), min_views={GROUND['min_views']})"
                                   if per_view else " (no calibrated view: max boxes in any view)"))

    metrics, gt_ground, label_pts = {}, None, {}
    frames = {v["frame"] for v in views if v["frame"] is not None}
    if LABELS and len(frames) == 1:
        fid = frames.pop()
        h5 = _p(LABELS.get("gt_h5"))
        if h5 and os.path.exists(h5):
            gt_ground = gt_points_for_frame(load_gt_groundplane(h5), fid)
        tp = fp = fn = 0
        for v in views:
            if v["idx"] is None:
                continue
            via = os.path.join(_p(LABELS["via_dir"]), f"via_region_data_view{v['idx']}.json")
            pts = load_via_points(via).get(fid) if os.path.exists(via) else None
            if pts is not None and len(pts):
                label_pts[v["id"]] = pts
                m = eval_view(v["boxes"], pts)
                tp, fp, fn = tp + m["tp"], fp + m["fp"], fn + m["fn"]
        if label_pts:
            p, r, f1 = prf(tp, fp, fn)
            metrics.update(img_precision=round(p, 3), img_recall=round(r, 3), img_f1=round(f1, 3))
            print(f"  image-space vs VIA labels: P {p:.3f} R {r:.3f} F1 {f1:.3f}")
        if gt_ground is not None and len(gt_ground) and per_view:
            t_, f_, n_, d_ = match_points(fused, gt_ground[:, :2], LABELS.get("match_radius_m", 1.0))
            p, r, f1 = prf(t_, f_, n_)
            metrics.update(gt_count=len(gt_ground), count_error=count - len(gt_ground), gp_precision=round(p, 3),
                           gp_recall=round(r, 3), gp_f1=round(f1, 3),
                           gp_median_err_m=round(float(np.median(d_)), 3) if len(d_) else "")
            print(f"  ground-plane vs GT ({len(gt_ground)} people): P {p:.3f} R {r:.3f} F1 {f1:.3f}"
                  f" | count error {count - len(gt_ground):+d}")
    elif LABELS:
        print("  [info] evaluation skipped: image filenames must share one frame_XXXX id")

    overlays = [draw_view(v["img"], v["boxes"], label_pts.get(v["id"]), scale=0.5) for v in views]
    img_path = os.path.join(OUT_DIR, "result.png")
    plot_result(img_path, overlays, grid, fused, nviews, density, count, gt_ground,
                n_boxes=sum(len(v["boxes"]) for v in views))
    print(f"  -> {img_path}")

    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "cameras": len(views),
           "model": YOLO["model"], "conf": YOLO["conf"], "persons": count, "calibrated_views": len(per_view),
           **{f"boxes_{v['id']}": len(v["boxes"]) for v in views}, **metrics,
           "elapsed_s": round(time.time() - t0, 2)}
    print(f"  -> {_write_csv(row)}")
    print(f"Done in {row['elapsed_s']}s")


if __name__ == "__main__":
    main()