#!/usr/bin/env python3
"""yolo.py: YOLO person detection for CityStreet (tiled, evaluated, ground-plane fused).

Press Play in VS Code to run the 'eval' command with the defaults in the SETTINGS block below.
Command-line use still works:

    python yolo.py eval --frames 636,700,800
    python yolo.py eval --frames 636 --model yolo11m.pt
    python yolo.py image path\\to\\a.jpg path\\to\\b.jpg --conf 0.25

Needs crowd3d/crowd3d_all.py (imports Camera, CityStreetDataset, DatasetPaths from it).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# --- SETTINGS (edit these, then press Play) ---
HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "CityStreet"
OUTPUT_DIR = HERE / "yolo_outputs"
MODEL = "yolo11x.pt"
DEFAULT_FRAMES = "636"

for _p in (HERE, HERE / "crowd3d"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from crowd3d_all import Camera, CityStreetDataset, DatasetPaths


# --- 1. CONFIG ---
@dataclass
class YoloConfig:
    model: str = MODEL
    device: Optional[str] = None
    conf: float = 0.20
    iou: float = 0.55
    imgsz_full: int = 1280
    tile: int = 1024
    tile_overlap: float = 0.25
    tile_imgsz: int = 1024
    use_full: bool = True
    use_tiles: bool = True
    augment: bool = False
    max_det: int = 1500
    merge_iou: float = 0.50
    merge_ios: float = 0.85
    edge_px: int = 6
    min_box_h: float = 14.0
    aspect_min: float = 0.8
    aspect_max: float = 8.0


# --- 2. DETECTION ---
def make_tiles(h: int, w: int, tile: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    step = max(1, int(tile * (1.0 - overlap)))

    def starts(n):
        if n <= tile:
            return [0]
        s = list(range(0, n - tile + 1, step))
        if s[-1] + tile < n:
            s.append(n - tile)
        return s

    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in starts(h) for x in starts(w)]


def drop_cut_boxes(b: np.ndarray, tile_box, img_wh, edge: int) -> np.ndarray:
    x0, y0, x1, y1 = tile_box
    w, h = img_wh
    cut = ((b[:, 0] <= x0 + edge) & (x0 > 0)) | ((b[:, 2] >= x1 - edge) & (x1 < w)) \
        | ((b[:, 1] <= y0 + edge) & (y0 > 0)) | ((b[:, 3] >= y1 - edge) & (y1 < h))
    return ~cut


def merge_boxes(b: np.ndarray, s: np.ndarray, iou_thr: float, ios_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(b) == 0:
        return b, s
    order = np.argsort(-s, kind="stable")
    b, s = b[order], s[order]
    area = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    alive = np.ones(len(b), dtype=bool)
    keep = []
    for i in range(len(b)):
        if not alive[i]:
            continue
        keep.append(i)
        if i + 1 >= len(b):
            break
        r = slice(i + 1, None)
        iw = np.maximum(np.minimum(b[i, 2], b[r, 2]) - np.maximum(b[i, 0], b[r, 0]), 0)
        ih = np.maximum(np.minimum(b[i, 3], b[r, 3]) - np.maximum(b[i, 1], b[r, 1]), 0)
        inter = iw * ih
        iou = inter / (area[i] + area[r] - inter + 1e-9)
        ios = inter / (area[r] + 1e-9)
        alive[r] &= ~((iou > iou_thr) | (ios > ios_thr))
    return b[keep], s[keep]


def filter_boxes(b: np.ndarray, s: np.ndarray, cfg: YoloConfig, roi: Optional[np.ndarray] = None):
    if len(b) == 0:
        return b, s
    w = np.maximum(b[:, 2] - b[:, 0], 1e-3)
    h = np.maximum(b[:, 3] - b[:, 1], 1e-3)
    ok = (h >= cfg.min_box_h) & (h / w >= cfg.aspect_min) & (h / w <= cfg.aspect_max)
    if roi is not None:
        H, W = roi.shape[:2]
        fx = np.clip(((b[:, 0] + b[:, 2]) / 2).astype(int), 0, W - 1)
        fy = np.clip(b[:, 3].astype(int), 0, H - 1)
        ok &= roi[fy, fx] > 0
    return b[ok], s[ok]


class PersonDetector:
    def __init__(self, cfg: YoloConfig):
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError:
            sys.exit("ultralytics is not installed in the Python you are running.\n"
                     "Install it with:  python -m pip install ultralytics\n"
                     "and make sure VS Code uses that interpreter (Ctrl+Shift+P > Python: Select Interpreter).")
        self.cfg = cfg
        self.model = YOLO(cfg.model)

    def _predict(self, img: np.ndarray, imgsz: int) -> Tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        r = self.model.predict(img, imgsz=imgsz, conf=c.conf, iou=c.iou, classes=[0], device=c.device,
                               augment=c.augment, max_det=c.max_det, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        return r.boxes.xyxy.cpu().numpy().astype(np.float32), r.boxes.conf.cpu().numpy().astype(np.float32)

    def detect(self, img_bgr: np.ndarray, roi: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        h, w = img_bgr.shape[:2]
        B, S = [], []
        if c.use_full:
            b, s = self._predict(img_bgr, c.imgsz_full)
            B.append(b), S.append(s)
        if c.use_tiles:
            for (x0, y0, x1, y1) in make_tiles(h, w, c.tile, c.tile_overlap):
                b, s = self._predict(img_bgr[y0:y1, x0:x1], c.tile_imgsz)
                if len(b) == 0:
                    continue
                b = b + np.array([x0, y0, x0, y0], np.float32)
                k = drop_cut_boxes(b, (x0, y0, x1, y1), (w, h), c.edge_px)
                B.append(b[k]), S.append(s[k])
        if not B:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        b, s = merge_boxes(np.concatenate(B), np.concatenate(S), c.merge_iou, c.merge_ios)
        return filter_boxes(b, s, c, roi)


# --- 3. GROUND PLANE ---
def boxes_to_ground(cam: Camera, boxes: np.ndarray, mode: str = "head", head_h: float = 1.70,
                    bounds=(-30.0, 35.0, -40.0, 35.0)) -> Tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    if mode == "feet":
        uv, z = np.column_stack([cx, boxes[:, 3]]), 0.0
    else:
        uv, z = np.column_stack([cx, boxes[:, 1]]), head_h
    pts, ok = cam.backproject_ground(uv, z)
    xy = pts[:, :2]
    ok &= (xy[:, 0] > bounds[0]) & (xy[:, 0] < bounds[1]) & (xy[:, 1] > bounds[2]) & (xy[:, 1] < bounds[3])
    return xy, ok


def fuse_views(per_view: Dict[str, Tuple[np.ndarray, np.ndarray]], radius: float = 1.0):
    ids = {v: i for i, v in enumerate(per_view)}
    P, W, M, S = [], [], [], []
    for v, (xy, sc) in per_view.items():
        for p, s in zip(xy, sc):
            P.append(p), W.append(float(s)), M.append(1 << ids[v]), S.append([float(s)])
    if not P:
        return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0)
    C, W, M = np.array(P, float), np.array(W), np.array(M, dtype=np.int64)
    alive = np.ones(len(C), dtype=bool)
    while True:
        idx = np.flatnonzero(alive)
        if len(idx) < 2:
            break
        c = C[idx]
        D = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
        D[(M[idx][:, None] & M[idx][None, :]) != 0] = np.inf
        a, b = np.unravel_index(np.argmin(D), D.shape)
        if not np.isfinite(D[a, b]) or D[a, b] > radius:
            break
        i, j = idx[a], idx[b]
        C[i] = (C[i] * W[i] + C[j] * W[j]) / (W[i] + W[j])
        W[i] += W[j]
        M[i] |= M[j]
        S[i] += S[j]
        alive[j] = False
    idx = np.flatnonzero(alive)
    nv = np.array([bin(int(M[i])).count("1") for i in idx])
    return C[idx], nv, np.array([np.mean(S[i]) for i in idx])


def match_points(pred: np.ndarray, gt: np.ndarray, radius: float):
    if len(pred) == 0 or len(gt) == 0:
        return 0, len(pred), len(gt), np.zeros(0)
    D = np.hypot(pred[:, None, 0] - gt[None, :, 0], pred[:, None, 1] - gt[None, :, 1])
    r, c = linear_sum_assignment(D)
    good = D[r, c] <= radius
    tp = int(good.sum())
    return tp, len(pred) - tp, len(gt) - tp, D[r, c][good]


# --- 4. IMAGE-SPACE EVALUATION (VIA point labels) ---
def match_boxes_to_points(boxes: np.ndarray, pts: np.ndarray, margin: float = 0.10):
    n, m = len(boxes), len(pts)
    if n == 0 or m == 0:
        return []
    x0, y0, x1, y1 = boxes.T
    w, h = x1 - x0, y1 - y0
    inside = ((pts[None, :, 0] >= (x0 - margin * w)[:, None]) & (pts[None, :, 0] <= (x1 + margin * w)[:, None])
              & (pts[None, :, 1] >= (y0 - margin * h)[:, None]) & (pts[None, :, 1] <= (y1 + margin * h)[:, None]))
    rx, ry = (x0 + x1) / 2, y0 + 0.2 * h
    d = np.hypot(pts[None, :, 0] - rx[:, None], pts[None, :, 1] - ry[:, None]) / np.maximum(h[:, None], 1.0)
    cost = np.where(inside, d, 1e6)
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] < 1e5
    return list(zip(r[ok], c[ok]))


def eval_view(boxes: np.ndarray, pts: np.ndarray) -> Dict[str, float]:
    m = match_boxes_to_points(boxes, pts)
    tp = len(m)
    rel = [(pts[j, 1] - boxes[i, 1]) / max(boxes[i, 3] - boxes[i, 1], 1.0) for i, j in m]
    return {"tp": tp, "fp": len(boxes) - tp, "fn": len(pts) - tp,
            "rel_y": float(np.median(rel)) if rel else float("nan")}


def prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# --- 5. VISUALS ---
def draw_view(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, pts: Optional[np.ndarray] = None,
              scale: float = 0.5) -> np.ndarray:
    out = img.copy()
    for (x0, y0, x1, y1), s in zip(boxes, scores):
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 220, 0), 2)
    if pts is not None:
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), 7, (0, 0, 255), 2)
    return cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def plot_bev(path: str, fused: np.ndarray, nviews: np.ndarray, gt: Optional[np.ndarray], title: str):
    fig, ax = plt.subplots(figsize=(8, 9))
    if gt is not None and len(gt):
        ax.scatter(gt[:, 0], gt[:, 1], marker="x", c="#ff0055", s=40, label=f"GT ({len(gt)})")
    if len(fused):
        sc = ax.scatter(fused[:, 0], fused[:, 1], c=nviews, cmap="viridis", s=26, edgecolors="k", linewidths=0.4,
                        vmin=1, vmax=3, label=f"YOLO fused ({len(fused)})")
        plt.colorbar(sc, ax=ax, label="views supporting")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m, down)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --- 6. EVAL COMMAND ---
def pick_frames(ds: CityStreetDataset, arg: str, n: int) -> List[int]:
    if arg:
        if ":" in arg:
            a = [int(x) for x in arg.split(":")]
            return list(range(a[0], a[1] + 1, a[2] if len(a) > 2 else 2))
        return [int(x) for x in arg.split(",") if x.strip()]
    sets = [set(ds.load_via_points(v).keys()) for v in ds.cameras]
    common = sorted(set.intersection(*sets) & set(int(f) for f in ds.gt_frame_ids()))
    return common[:: max(1, len(common) // max(n, 1))][:n]


def cmd_eval(a):
    cfg = YoloConfig(model=a.model, device=a.device, conf=a.sweep_min, imgsz_full=a.imgsz, tile=a.tile,
                     tile_overlap=a.overlap, tile_imgsz=a.tile_imgsz, use_full=not a.no_full,
                     use_tiles=not a.no_tiles, augment=a.augment)
    ds = CityStreetDataset(DatasetPaths(root_dir=a.data_root))
    ds.verify_paths()
    if not os.path.isdir(ds.paths.images_dir):
        sys.exit(f"\nImages folder not found: {ds.paths.images_dir}\nFix DATA_ROOT at the top of yolo.py.")
    frames = pick_frames(ds, a.frames, a.num_frames)
    if not frames:
        sys.exit("\nNo frames selected. Set DEFAULT_FRAMES at the top of yolo.py, e.g. \"636\".")
    print(f"frames: {frames} | model {cfg.model} | full={cfg.use_full}@{cfg.imgsz_full} "
          f"tiles={cfg.use_tiles}@{cfg.tile}/{cfg.tile_overlap}")
    det = PersonDetector(cfg)
    os.makedirs(a.output_dir, exist_ok=True)
    thr_list = [round(t, 2) for t in np.arange(a.sweep_min, 0.71, 0.05)]
    cache, t0 = {}, time.time()
    for fid in frames:
        imgs = ds.load_frame_images(fid)
        for v, im in imgs.items():
            t1 = time.time()
            cache[(fid, v)] = det.detect(im, ds.camera_roi_masks.get(v))
            print(f"  frame {fid} {v}: {len(cache[(fid, v)][0])} boxes @conf>={cfg.conf} ({time.time() - t1:.1f}s)")
    print(f"detection total {time.time() - t0:.1f}s")

    labels = {(f, v): ds.load_via_points(v).get(f) for f in frames for v in ds.cameras}
    have = {k: p for k, p in labels.items() if p is not None and len(p)}
    rows, best = [], (-1.0, a.sweep_min)
    if have:
        print("\nIMAGE-SPACE EVALUATION vs VIA labels (all views, all frames)")
        print(" conf   boxes  labels    TP    FP    FN   Prec  Rec    F1")
        for t in thr_list:
            tp = fp = fn = nb = 0
            for k, pts in have.items():
                b, s = cache[k]
                bt = b[s >= t]
                m = eval_view(bt, pts)
                tp, fp, fn, nb = tp + m["tp"], fp + m["fp"], fn + m["fn"], nb + len(bt)
            p, r, f1 = prf(tp, fp, fn)
            rows.append((t, nb, tp + fn, tp, fp, fn, p, r, f1))
            best = max(best, (f1, t))
            print(f" {t:4.2f}  {nb:6d} {tp + fn:7d} {tp:5d} {fp:5d} {fn:5d}  {p:.3f} {r:.3f} {f1:.3f}")
        thr = best[1]
        print(f"\nbest F1 {best[0]:.3f} at conf {thr}")
        rel = [eval_view(cache[k][0][cache[k][1] >= thr], pts)["rel_y"] for k, pts in have.items()]
        print(f"median label position inside matched boxes (0=top, 1=bottom): {np.nanmedian(rel):.2f}"
              "  (~0.1-0.2 = head labels, ~0.5 = body-centre labels)")
        for v in ds.cameras:
            ks = [k for k in have if k[1] == v]
            tp = sum(eval_view(cache[k][0][cache[k][1] >= thr], have[k])["tp"] for k in ks)
            ntot = sum(len(have[k]) for k in ks)
            print(f"  {v}: recall {tp / max(ntot, 1):.2%} ({tp}/{ntot})")
        with open(os.path.join(a.output_dir, "yolo_conf_sweep.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["conf", "boxes", "labels", "tp", "fp", "fn", "precision", "recall", "f1"])
            w.writerows(rows)
    else:
        thr = a.conf if a.conf is not None else 0.25
        print("\nno VIA labels for these frames; using conf", thr)
    thr = a.conf if a.conf is not None else thr

    print(f"\nGROUND-PLANE EVALUATION (conf >= {thr}, fuse radius {a.fuse_radius} m, GT match radius {a.match_radius} m)")
    print("  projection  min_views |   TP   FP   FN   Prec  Rec    F1 | count MAE | median loc err")
    for mode in ("head", "feet"):
        acc = {mv: [0, 0, 0, [], []] for mv in (1, 2)}
        for fid in frames:
            gt = ds.load_gt_groundplane_points(fid)
            if not len(gt):
                continue
            per = {}
            for v, cam in ds.cameras.items():
                b, s = cache[(fid, v)]
                b, s = b[s >= thr], s[s >= thr]
                xy, ok = boxes_to_ground(cam, b, mode)
                per[v] = (xy[ok], s[ok])
            fused, nv, _ = fuse_views(per, a.fuse_radius)
            for mv, r in acc.items():
                keep = nv >= mv
                t_, f_, n_, d_ = match_points(fused[keep], gt[:, :2], a.match_radius)
                r[0], r[1], r[2] = r[0] + t_, r[1] + f_, r[2] + n_
                r[3] += list(d_)
                r[4].append(abs(int(keep.sum()) - len(gt)))
                if mode == "head" and mv == a.min_views and fid == frames[0]:
                    plot_bev(os.path.join(a.output_dir, f"bev_{fid}.png"), fused[keep], nv[keep], gt[:, :2],
                             f"frame {fid}: YOLO fused (min_views={mv}) vs GT")
        for mv, r in acc.items():
            p, rc, f1 = prf(r[0], r[1], r[2])
            mae = np.mean(r[4]) if r[4] else float("nan")
            print(f"  {mode:10s}  {mv:9d} | {r[0]:4d} {r[1]:4d} {r[2]:4d}  {p:.3f} {rc:.3f} {f1:.3f} | "
                  f"{mae:9.1f} | {np.median(r[3]) if r[3] else float('nan'):.2f} m")

    for fid in frames[: a.save_frames]:
        imgs = ds.load_frame_images(fid)
        for v, im in imgs.items():
            b, s = cache[(fid, v)]
            vis = draw_view(im, b[s >= thr], s[s >= thr], labels.get((fid, v)))
            cv2.imwrite(os.path.join(a.output_dir, f"yolo_{fid}_{v}.jpg"), vis)
    print(f"outputs in {a.output_dir}")


# --- 7. IMAGE COMMAND ---
def cmd_image(a):
    if not a.images:
        sys.exit("Give at least one image path, e.g.  python yolo.py image photo.jpg")
    cfg = YoloConfig(model=a.model, device=a.device, conf=a.conf, imgsz_full=a.imgsz, tile=a.tile,
                     tile_overlap=a.overlap, tile_imgsz=a.tile_imgsz, use_full=not a.no_full, use_tiles=not a.no_tiles)
    det = PersonDetector(cfg)
    os.makedirs(a.output_dir, exist_ok=True)
    for p in a.images:
        img = cv2.imread(p)
        if img is None:
            print("cannot read", p)
            continue
        t0 = time.time()
        b, s = det.detect(img)
        out = os.path.join(a.output_dir, "yolo_" + os.path.basename(p))
        cv2.imwrite(out, draw_view(img, b, s, None, scale=min(1.0, 1600.0 / max(img.shape[:2]))))
        print(f"{p}: {len(b)} persons ({time.time() - t0:.1f}s) -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="YOLO person detection for CityStreet")
    ap.add_argument("command", nargs="?", default="eval", choices=["eval", "image"])
    ap.add_argument("images", nargs="*")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--frames", default=DEFAULT_FRAMES)
    ap.add_argument("--num-frames", type=int, default=3)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--tile-imgsz", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--no-full", action="store_true")
    ap.add_argument("--no-tiles", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--conf", type=float, default=None, help="final threshold (default: best F1 from sweep)")
    ap.add_argument("--sweep-min", type=float, default=0.10)
    ap.add_argument("--min-views", type=int, default=1)
    ap.add_argument("--fuse-radius", type=float, default=1.0)
    ap.add_argument("--match-radius", type=float, default=1.0)
    ap.add_argument("--save-frames", type=int, default=1)
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    a = ap.parse_args(argv)
    if a.command == "image":
        a.conf = 0.25 if a.conf is None else a.conf
        cmd_image(a)
    else:
        cmd_eval(a)


if __name__ == "__main__":
    main()