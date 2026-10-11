"""evaluate_yolo.py: how well does YOLO detect people on CityStreet?

Measures, against the dataset's own labels:
  1. image space   detections vs the per-view VIA point labels (P/R/F1, AP, recall by person size/distance/camera)
  2. ground plane  fused detections vs ground-truth positions (P/R/F1, count MAE/RMSE/bias, localisation error,
                   recall vs number of cameras that can see the person)
  3. optional      --compare: full-frame vs tiled inference
The confidence threshold is chosen on train-range frames (636-1234) and reported on held-out frames (1236-1634).
Detections are cached in eval_cache/, so re-runs and threshold changes do not re-run YOLO.

Usage:
    .\\.crowd\\Scripts\\python.exe evaluate_yolo.py
    .\\.crowd\\Scripts\\python.exe evaluate_yolo.py --n-tune 6 --n-test 12 --device 0
    .\\.crowd\\Scripts\\python.exe evaluate_yolo.py --compare
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crowd3d.crowd3d_all import (
    CityStreetData, PersonDetector, YoloConfig, boxes_to_ground, fuse_views, match_boxes_to_points,
    fuse_pixels, groundplane_px_to_world_m, refine_camera, GroundGrid, ground_density,
)

DATA_ROOT = "CityStreet"
OUT_DIR = "eval_outputs"
CACHE_DIR = "eval_cache"
YOLO = dict(model="yolo11x.pt", device=None, imgsz_full=1280, use_full=True, use_tiles=True,
            tile=1024, tile_overlap=0.25, tile_imgsz=1024)
SWEEP_MIN = 0.05
THRESHOLDS = [round(float(t), 2) for t in np.arange(SWEEP_MIN, 0.71, 0.05)]
SIZE_BINS = [0, 40, 60, 90, 140, 1e9]
HEAD_H = 1.70
FUSE_RADIUS = 1.0
MATCH_RADIUS = 1.0
ASSOC_TOL = 0.4
REGION_PAD = 1.0
TUNE_RANGE = (636, 1234)
TEST_RANGE = (1236, 1634)
STRATEGIES = {
    "full @640": dict(use_full=True, use_tiles=False, imgsz_full=640),
    "full @1280": dict(use_full=True, use_tiles=False, imgsz_full=1280),
    "tiles only": dict(use_full=False, use_tiles=True),
    "full @1280 + tiles (default)": dict(),
}


def spread(items, n):
    items = list(items)
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return items
    return [items[int(i)] for i in np.linspace(0, len(items) - 1, n).round()]


def pick_frames(ds, args):
    via = [set(ds.via_points(v).keys()) for v in ds.cameras]
    gt_ids = {int(f) for f in np.unique(ds.gt()[:, 0])} if len(ds.gt()) else set()
    ok = sorted(set.intersection(*via) & gt_ids)
    exists = lambda f: os.path.exists(os.path.join(ds.images_dir, "camera1", f"frame_{f:04d}.jpg"))
    ok = [f for f in ok if exists(f)]

    def parse(s):
        return [int(x) for x in s.split(",") if x.strip()]

    tune = parse(args.tune_frames) if args.tune_frames else spread([f for f in ok if TUNE_RANGE[0] <= f <= TUNE_RANGE[1]], args.n_tune)
    test = parse(args.test_frames) if args.test_frames else spread([f for f in ok if TEST_RANGE[0] <= f <= TEST_RANGE[1]], args.n_test)
    return tune, test, ok


def cfg_tag(cfg):
    s = (f"{os.path.splitext(os.path.basename(cfg.model))[0]}_f{int(cfg.use_full)}-{cfg.imgsz_full}"
         f"_t{int(cfg.use_tiles)}-{cfg.tile}-{cfg.tile_overlap}-{cfg.tile_imgsz}_c{cfg.conf}_h{cfg.min_box_h}")
    return s.replace(".", "p")


def detect_all(ds, cfg, frames, use_cache):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag, det, out, times = cfg_tag(cfg), None, {}, []
    for fid in frames:
        imgs = None
        for v in ds.cameras:
            p = os.path.join(CACHE_DIR, f"{tag}_{fid}_{v}.npz")
            if use_cache and os.path.exists(p):
                d = np.load(p)
                out[(fid, v)] = (d["boxes"], d["scores"])
                continue
            det = det or PersonDetector(cfg)
            imgs = imgs or ds.load_frame_images(fid)
            t0 = time.time()
            b, s = det.detect(imgs[v], ds.roi.get(v))
            times.append(time.time() - t0)
            np.savez(p, boxes=b, scores=s)
            out[(fid, v)] = (b, s)
            print(f"  frame {fid} {v}: {len(b)} boxes ({times[-1]:.1f}s)", flush=True)
    return out, (float(np.mean(times)) if len(times) == len(frames) * len(ds.cameras) else None)


def label_geometry(cam, pts):
    n = len(pts)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    G, ok = cam.backproject_ground(pts, HEAD_H)
    xy = G[:, :2]
    feet, _ = cam.project(np.column_stack([xy, np.zeros(n)]), check_bounds=False)
    head, _ = cam.project(np.column_stack([xy, np.full(n, HEAD_H)]), check_bounds=False)
    h = feet[:, 1] - head[:, 1]
    d = np.hypot(xy[:, 0] - cam.center_m[0], xy[:, 1] - cam.center_m[1])
    bad = ~ok | ~np.isfinite(h) | (h <= 0) | (h > 1500)
    return np.where(bad, np.nan, h), np.where(bad, np.nan, d)


def build_labels(ds, frames):
    L = {}
    for fid in frames:
        for v, cam in ds.cameras.items():
            pts = ds.via_points(v).get(fid)
            pts = np.zeros((0, 2)) if pts is None else pts
            h, d = label_geometry(cam, pts)
            roi = ds.roi.get(v)
            if roi is None or not len(pts):
                inroi = np.ones(len(pts), bool)
            else:
                inroi = roi[np.clip(pts[:, 1].astype(int), 0, roi.shape[0] - 1),
                            np.clip(pts[:, 0].astype(int), 0, roi.shape[1] - 1)] > 0
            L[(fid, v)] = dict(pts=pts, h=h, d=d, inroi=inroi)
    return L


def image_metrics(dets, L, keys, t, detail=False):
    tp = fp = fn = 0
    rows = []
    for k in keys:
        b, s = dets[k]
        sel = s >= t
        bt, lab = b[sel], L[k]
        m = match_boxes_to_points(bt, lab["pts"])
        tp, fp, fn = tp + len(m), fp + len(bt) - len(m), fn + len(lab["pts"]) - len(m)
        if detail:
            hit = np.zeros(len(lab["pts"]), bool)
            for _, j in m:
                hit[j] = True
            rows.append((k, bt, m, hit))
    return (tp, fp, fn, rows) if detail else (tp, fp, fn)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def sweep(dets, L, keys):
    out = []
    for t in THRESHOLDS:
        tp, fp, fn = image_metrics(dets, L, keys, t)
        p, r, f1 = prf(tp, fp, fn)
        out.append(dict(conf=t, boxes=tp + fp, labels=tp + fn, tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1))
    return out


def average_precision(rows):
    pts = sorted((r["r"], r["p"]) for r in rows)
    if len(pts) < 2:
        return float("nan")
    R, P = np.array([x[0] for x in pts]), np.array([x[1] for x in pts])
    return float(np.sum(np.diff(R) * (P[1:] + P[:-1]) / 2.0) + R[0] * P[0])


def recall_by(values, hits, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = np.isfinite(values) & (values >= lo) & (values < hi)
        out.append((lo, hi, int(m.sum()), float(hits[m].mean()) if m.any() else float("nan")))
    return out


def collect_label_hits(dets, L, keys, t):
    _, _, _, rows = image_metrics(dets, L, keys, t, detail=True)
    H, D, hit, view, inroi = [], [], [], [], []
    for k, _, _, h in rows:
        H += list(L[k]["h"]); D += list(L[k]["d"]); hit += list(h); view += [k[1]] * len(h); inroi += list(L[k]["inroi"])
    return np.array(H), np.array(D), np.array(hit, bool), np.array(view), np.array(inroi, bool)


def near_miss_stats(rows, L, frac=0.75):
    near = total = 0
    for k, bt, m, hit in rows:
        matched = {i for i, _ in m}
        un = [i for i in range(len(bt)) if i not in matched]
        for j in np.flatnonzero(~hit):
            total += 1
            if un:
                u = bt[un]
                h = np.maximum(u[:, 3] - u[:, 1], 1.0)
                d = np.hypot(L[k]["pts"][j, 0] - (u[:, 0] + u[:, 2]) / 2, L[k]["pts"][j, 1] - (u[:, 1] + 0.2 * h)) / h
                near += int((d < frac).any())
    return near, total


def match_idx(pred, gt, radius):
    if len(pred) == 0 or len(gt) == 0:
        return set(), set(), []
    D = np.hypot(pred[:, None, 0] - gt[None, :, 0], pred[:, None, 1] - gt[None, :, 1])
    r, c = linear_sum_assignment(D)
    good = D[r, c] <= radius
    return set(c[good].tolist()), set(r[good].tolist()), D[r, c][good].tolist()


def visible_cams(ds, gt):
    n = np.zeros(len(gt), int)
    for v, cam in ds.cameras.items():
        uv0, ok0 = cam.project(np.column_stack([gt[:, :2], np.zeros(len(gt))]))
        _, okh = cam.project(np.column_stack([gt[:, :2], gt[:, 2]]))
        ok = ok0 & okh
        roi = ds.roi.get(v)
        if roi is not None and ok.any():
            u = np.clip(uv0[:, 0].astype(int), 0, roi.shape[1] - 1)
            w = np.clip(uv0[:, 1].astype(int), 0, roi.shape[0] - 1)
            ok &= roi[w, u] > 0
        n += ok
    return n


def gt_region(ds, pad=REGION_PAD):
    g = ds.gt()
    if not len(g):
        return None
    X, Y = groundplane_px_to_world_m(g[:, 2], g[:, 3])
    return (float(X.min()) - pad, float(X.max()) + pad, float(Y.min()) - pad, float(Y.max()) + pad)


def in_region(xy, reg):
    if reg is None or not len(xy):
        return np.ones(len(xy), bool)
    return (xy[:, 0] >= reg[0]) & (xy[:, 0] <= reg[1]) & (xy[:, 1] >= reg[2]) & (xy[:, 1] <= reg[3])


def box_top_pixels(b):
    if not len(b):
        return np.zeros((0, 2))
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 1]])


def ground_eval(ds, dets, frames, thr, mode, mvs=(1, 2), clip=True):
    res = {mv: dict(tp=0, fp=0, fn=0, d=[], err=[], gt=[], pred=[]) for mv in mvs}
    gt_vis = []
    reg = gt_region(ds) if clip else None
    for fid in frames:
        gt = ds.gt_points(fid)
        if not len(gt):
            continue
        if mode == "tri":
            uvs, scs = {}, {}
            for v in ds.cameras:
                b, s = dets[(fid, v)]
                sel = s >= thr
                uvs[v], scs[v] = box_top_pixels(b[sel]), s[sel]
            fused, nv, _, _ = fuse_pixels(ds.cameras, uvs, scs, ASSOC_TOL)
        else:
            per = {}
            for v, cam in ds.cameras.items():
                b, s = dets[(fid, v)]
                sel = s >= thr
                xy, ok = boxes_to_ground(cam, b[sel], mode, HEAD_H)
                per[v] = (xy[ok], s[sel][ok])
            fused, nv, _ = fuse_views(per, FUSE_RADIUS)
        keep = in_region(fused, reg)
        fused, nv = fused[keep], nv[keep]
        nc = visible_cams(ds, gt)
        for mv in mvs:
            pred = fused[nv >= mv]
            mg, _, d = match_idx(pred, gt[:, :2], MATCH_RADIUS)
            r = res[mv]
            r["tp"] += len(mg); r["fp"] += len(pred) - len(mg); r["fn"] += len(gt) - len(mg)
            r["d"] += d; r["err"].append(len(pred) - len(gt)); r["gt"].append(len(gt)); r["pred"].append(len(pred))
            if mv == 1:
                gt_vis += [(int(nc[i]), i in mg) for i in range(len(gt))]
    return res, gt_vis


def score_points(pred, gt):
    mg, _, d = match_idx(pred, gt, MATCH_RADIUS)
    return len(mg), len(pred) - len(mg), len(gt) - len(mg)


def oracle_report(ds, L, frames):
    reg = gt_region(ds)
    print(f"\nORACLE FUSION: ground-truth head labels used as detections (frames {frames}, match {MATCH_RADIUS} m)")
    print("  method              min_views | pred/frame  GT/frame |   TP   FP   FN   Prec   Rec")
    methods = [(f"radius {r} m", ("radius", r)) for r in (0.5, 1.0, 1.5, 2.0)] + [(f"triangulate {ASSOC_TOL} m", ("tri", ASSOC_TOL))]
    for name, (kind, par) in methods:
        acc = {1: np.zeros(5), 2: np.zeros(5)}
        for fid in frames:
            gt = ds.gt_points(fid)
            if not len(gt):
                continue
            if kind == "radius":
                per = {}
                for v, cam in ds.cameras.items():
                    G, ok = cam.backproject_ground(L[(fid, v)]["pts"], HEAD_H)
                    per[v] = (G[ok, :2], np.ones(int(ok.sum())))
                fused, nv, _ = fuse_views(per, par)
            else:
                uvs = {v: L[(fid, v)]["pts"] for v in ds.cameras}
                fused, nv, _, _ = fuse_pixels(ds.cameras, uvs, None, par)
            keep = in_region(fused, reg)
            fused, nv = fused[keep], nv[keep]
            for mv in (1, 2):
                p = fused[nv >= mv]
                tp, fp, fn = score_points(p, gt[:, :2])
                acc[mv] += np.array([len(p), len(gt), tp, fp, fn])
        for mv, a in acc.items():
            n = max(len(frames), 1)
            p, r, _ = prf(a[2], a[3], a[4])
            print(f"  {name:19s} {mv:9d} | {a[0] / n:10.1f} {a[1] / n:9.1f} | {int(a[2]):4d} {int(a[3]):4d} {int(a[4]):4d}  {p:.3f} {r:.3f}")


def reproj_report(ds, L, frames, gate=80.0):
    bands = [(0, 600), (600, 800), (800, 1000), (1000, 4000)]
    print(f"\nREPROJECTION CHECK: ground-truth 3D head points projected into each camera vs the image labels (frames {frames}, gate {gate:.0f} px)")
    print("  view | GT in image  matched  <=8 px | median err  p90 err | mean dx  mean dy | median err by image row: " + "  ".join(f"{a}-{b}" for a, b in bands))
    for v, cam in ds.cameras.items():
        errs, dxy, rows = [], [], []
        total = 0
        for fid in frames:
            gt = ds.gt_points(fid)
            if not len(gt):
                continue
            uv, ok = cam.project(gt)
            uv = uv[ok]
            total += len(uv)
            lab = np.asarray(L[(fid, v)]["pts"], float).reshape(-1, 2)
            if not len(uv) or not len(lab):
                continue
            D = np.hypot(uv[:, None, 0] - lab[None, :, 0], uv[:, None, 1] - lab[None, :, 1])
            a, b = linear_sum_assignment(D)
            good = D[a, b] <= gate
            errs += list(D[a, b][good])
            dxy += list((lab[b[good]] - uv[a[good]]))
            rows += list(uv[a[good], 1])
        errs, rows = np.array(errs), np.array(rows)
        dxy = np.array(dxy).reshape(-1, 2)
        if not len(errs):
            print(f"  {v:5s}| no matches")
            continue
        band_txt = "  ".join(
            (f"{np.median(errs[(rows >= a) & (rows < b)]):7.1f}" if ((rows >= a) & (rows < b)).sum() > 5 else "      -")
            for a, b in bands)
        print(f"  {v:5s}| {total:11d} {len(errs) / max(total, 1):8.1%} {(errs <= 8).sum() / max(total, 1):7.1%} | {np.median(errs):10.1f} {np.percentile(errs, 90):8.1f} | "
              f"{dxy[:, 0].mean():+7.1f} {dxy[:, 1].mean():+8.1f} | {band_txt}")


def block_counts(d, grid, size_m):
    b = int(round(size_m / grid.cell))
    ny, nx = d.shape
    d = d[:ny // b * b, :nx // b * b]
    return d.reshape(ny // b, b, nx // b, b).sum(axis=(1, 3)) * grid.cell ** 2


def density_report(ds, dets, tune, test, thr, out_dir):
    grid = GroundGrid()
    reg = gt_region(ds)
    cell2 = grid.cell ** 2
    modes = {"head": ("top", HEAD_H), "mid": ("mid", 0.85), "feet": ("bottom", 0.0)}
    sigmas = (0.6, 1.0, 1.5, 2.0)
    views = list(ds.cameras)
    cache = {}

    def pts(fid, v, mode):
        key = (fid, v, mode)
        if key not in cache:
            b, s_ = dets[(fid, v)]
            b = b[s_ >= thr]
            if not len(b):
                cache[key] = np.zeros((0, 2))
            else:
                which, z = modes[mode]
                y = b[:, 1] if which == "top" else (b[:, 3] if which == "bottom" else (b[:, 1] + b[:, 3]) / 2)
                p, ok = ds.cameras[v].backproject_ground(np.column_stack([(b[:, 0] + b[:, 2]) / 2, y]), z)
                xy = p[:, :2][ok]
                cache[key] = xy[in_region(xy, reg)]
        return cache[key]

    scale = {}
    for mode in modes:
        for v in views:
            g = sum(len(ds.gt_points(f)) for f in tune)
            p = sum(len(pts(f, v, mode)) for f in tune)
            scale[(mode, v)] = g / p if p and g else 1.0

    xs = grid.x_min + (np.arange(grid.shape[1]) + 0.5) * grid.cell
    ys = grid.y_min + (np.arange(grid.shape[0]) + 0.5) * grid.cell
    mask = ((ys[:, None] >= reg[2]) & (ys[:, None] <= reg[3]) & (xs[None, :] >= reg[0]) & (xs[None, :] <= reg[1])).astype(np.float32)
    gt_maps = {(f, sg): ground_density(ds.gt_points(f)[:, :2], grid, sg) for f in test for sg in sigmas}
    gtc = float(np.mean([len(ds.gt_points(f)) for f in test]))

    def maps_for(mode, sg, source, f):
        vm = {v: ground_density(pts(f, v, mode), grid, sg) for v in views}
        if source in views:
            return vm[source]
        if source == "avg3":
            return sum(vm.values()) / len(views)
        return sum(scale[(mode, v)] * vm[v] for v in views) / len(views)

    def score(mode, sg, source):
        bias, e5, e10, P, G = [], [], [], [], []
        for f in test:
            p, g = maps_for(mode, sg, source, f), gt_maps[(f, sg)]
            bias.append((p.sum() - g.sum()) * cell2)
            b5p, b5g = block_counts(p, grid, 5), block_counts(g, grid, 5)
            e5.append(0.5 * np.abs(b5p - b5g).sum())
            e10.append(0.5 * np.abs(block_counts(p, grid, 10) - block_counts(g, grid, 10)).sum())
            P.append(b5p.ravel())
            G.append(b5g.ravel())
        corr = float(np.corrcoef(np.concatenate(P), np.concatenate(G))[0, 1])
        return float(np.mean(bias)), float(np.mean(e5)), float(np.mean(e10)), corr

    def line(label, r):
        b, e5, e10, c = r
        return f"{label} | {b:+6.1f} ({b / gtc:+5.0%}) | {e5:5.1f} ({e5 / gtc:4.0%})  corr {c:.2f} | {e10:5.1f} ({e10 / gtc:4.0%})"

    cams_model = next(iter(ds.cameras.values())).dist_model if hasattr(next(iter(ds.cameras.values())), "dist_model") else "n/a"
    print(f"\nGROUND-PLANE DENSITY MAPS from per-view detections (conf {thr}, test frames {test}, camera distortion model: {cams_model})")
    print(f"  average ground-truth count per frame: {gtc:.1f}. Each person is splatted as a Gaussian; per-view maps are averaged, so nobody has to be matched across cameras.")
    print("  block error = people sitting in the wrong block (half the summed absolute difference of block counts); lower is better; corr = correlation of 5 m block counts")
    print(f"  {'mode':5s} {'source':14s} | count bias per frame | 5 m block error        | 10 m block error   (kernel sigma 1.0 m)")
    results = {}
    for mode in modes:
        for source in views + ["avg3", "avg3_calibrated"]:
            r = score(mode, 1.0, source)
            results[(mode, 1.0, source)] = r
            print("  " + line(f"{mode:5s} {source:14s}", r))
    print("\n  averaged map (avg3) for different kernel widths:")
    print(f"  {'mode':5s} {'sigma':14s} | count bias per frame | 5 m block error        | 10 m block error")
    for mode in modes:
        for sg in sigmas:
            r = score(mode, sg, "avg3")
            results[(mode, sg, "avg3")] = r
            print("  " + line(f"{mode:5s} {sg:<14.1f}", r))
    uni = []
    for f in test:
        p = maps_for("head", 1.0, "avg3", f)
        u = mask * (p.sum() * cell2 / (mask.sum() * cell2))
        uni.append(0.5 * np.abs(block_counts(u, grid, 5) - block_counts(gt_maps[(f, 1.0)], grid, 5)).sum())
    print(f"  reference: spreading the same total evenly over the street gives a 5 m block error of {np.mean(uni):.1f} ({np.mean(uni) / gtc:.0%})")
    best = min(((k, v) for k, v in results.items() if k[2] in ("avg3", "avg3_calibrated")), key=lambda kv: kv[1][1])
    print(f"  best averaged map by 5 m block error: mode {best[0][0]}, sigma {best[0][1]}, {best[0][2]} -> {best[1][1]:.1f} people ({best[1][1] / gtc:.0%})")

    os.makedirs(out_dir, exist_ok=True)
    for f in test[:2]:
        fig, axes = plt.subplots(len(modes), len(views) + 2, figsize=(4 * (len(views) + 2), 3.6 * len(modes)))
        ext = (grid.x_min, grid.x_max, grid.y_max, grid.y_min)
        for i, mode in enumerate(modes):
            panels = [("ground truth", gt_maps[(f, 1.0)])] + [(v, maps_for(mode, 1.0, v, f)) for v in views] + [("average", maps_for(mode, 1.0, "avg3", f))]
            vmax = max(float(p.max()) for _, p in panels)
            for j, (name, p) in enumerate(panels):
                ax = axes[i, j]
                ax.imshow(p, extent=ext, vmin=0, vmax=vmax, cmap="magma")
                ax.set_title(f"{mode}: {name} ({p.sum() * cell2:.0f})", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"density_{f}.png"), dpi=90)
        plt.close(fig)
    print(f"  pictures saved to {out_dir}/density_<frame>.png (rows: head / mid / feet projection; columns: ground truth, each camera, average)")


LOS_EDGES = [0.31, 0.43, 0.72, 1.08, 2.17]
LOS_LETTERS = "ABCDEF"
OCCUPIED = 0.05


def los_letter(d):
    return LOS_LETTERS[int(np.searchsorted(LOS_EDGES, d, side="right"))]


def heatmap_report(ds, dets, tune, test, thr, out_dir, mode="auto", point="mid", abs_max=1.0, n_frames=2, sigma=1.0):
    from matplotlib import cm, colors
    import matplotlib.patches as mpatches
    grid = GroundGrid()
    reg = gt_region(ds)
    cell2 = grid.cell ** 2
    ny, nx = grid.shape
    which, z = {"head": ("top", HEAD_H), "mid": ("mid", 0.85), "feet": ("bottom", 0.0)}[point]
    views = list(ds.cameras)
    xs = grid.x_min + (np.arange(nx) + 0.5) * grid.cell
    ys = grid.y_min + (np.arange(ny) + 0.5) * grid.cell
    mask = (ys[:, None] >= reg[2]) & (ys[:, None] <= reg[3]) & (xs[None, :] >= reg[0]) & (xs[None, :] <= reg[1])
    area = mask.sum() * cell2
    ext = (grid.x_min, grid.x_max, grid.y_max, grid.y_min)
    cmap_abs = plt.get_cmap("turbo")
    rel_cols = [plt.get_cmap("YlOrRd")(x) for x in (0.12, 0.32, 0.52, 0.74, 0.96)]
    rel_edges = [0.5, 0.75, 0.9, 0.97]
    rel_names = ["lowest 50% of occupied area", "next 25%", "next 15%", "next 7%", "hottest 3%"]

    def view_pts(fid, v):
        b, s_ = dets[(fid, v)]
        b = b[s_ >= thr]
        if not len(b):
            return np.zeros((0, 2))
        y = b[:, 1] if which == "top" else (b[:, 3] if which == "bottom" else (b[:, 1] + b[:, 3]) / 2)
        p, ok = ds.cameras[v].backproject_ground(np.column_stack([(b[:, 0] + b[:, 2]) / 2, y]), z)
        xy = p[:, :2][ok]
        return xy[in_region(xy, reg)]

    scale = {}
    for v in views:
        g = sum(len(ds.gt_points(f)) for f in tune)
        p = sum(len(view_pts(f, v)) for f in tune)
        scale[v] = g / p if p and g else 1.0
    print(f"\nCROWD DENSITY HEATMAPS (people per m2) | projection point: {point} | kernel: each person spread over about a 2 m radius (sigma {sigma} m) | mode request: {mode}")
    print("  per-camera count correction (fitted on tune frames): " + ", ".join(f"{v} x{scale[v]:.2f}" for v in views))
    print("  reference levels (Fruin walkway level of service, people/m2): A <0.31 | B 0.31-0.43 | C 0.43-0.72 | D 0.72-1.08 | E 1.08-2.17 | F >2.17")
    print("  auto rule: if the busiest 10% of occupied area (p90) reaches 0.72 people/m2 (level D) the scene counts as crowded and the relative colour scale is used; otherwise the absolute scale.")

    def stats(d):
        occ = mask & (d >= OCCUPIED)
        vals = d[occ]
        out = dict(count=float((d * mask).sum() * cell2), occ_area=float(occ.sum() * cell2),
                   mean=float(vals.mean()) if vals.size else 0.0, p90=float(np.percentile(vals, 90)) if vals.size else 0.0,
                   peak=float((d * mask).max()))
        edges = [OCCUPIED] + LOS_EDGES + [1e9]
        out["shares"] = [float(((vals >= edges[i]) & (vals < edges[i + 1])).mean()) if vals.size else 0.0 for i in range(6)]
        return out

    def colorize(d, m, valid, alpha, ref):
        rgba = np.zeros(d.shape + (4,), np.float32)
        if m == "absolute":
            rgba[...] = cmap_abs(np.clip(d / abs_max, 0, 1))
            rgba[..., 3] = np.where(valid & (d >= 0.02), alpha, 0.0)
        else:
            occ = valid & (d >= OCCUPIED)
            if ref.size:
                r = np.searchsorted(ref, d, side="right") / ref.size
                band = np.digitize(r, rel_edges)
                for k in range(5):
                    rgba[band == k] = rel_cols[k]
            rgba[..., 3] = np.where(occ, alpha, 0.0)
        return rgba

    def ref_of(d):
        return np.sort(d[mask & (d >= OCCUPIED)])

    def overlay_samples(cam, dmap, stride=8):
        uu, vv = np.meshgrid(np.arange(0, cam.img_w, stride) + stride / 2, np.arange(0, cam.img_h, stride) + stride / 2)
        uv = np.column_stack([uu.ravel(), vv.ravel()])
        G, ok = cam.backproject_ground(uv, 0.0)
        with np.errstate(all="ignore"):
            ix = np.floor((G[:, 0] - grid.x_min) / grid.cell)
            iy = np.floor((G[:, 1] - grid.y_min) / grid.cell)
        inb = ok & np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix = np.where(inb, ix, 0).astype(int)
        iy = np.where(inb, iy, 0).astype(int)
        inb &= mask[iy, ix]
        d = np.where(inb, dmap[iy, ix], 0.0)
        return d.reshape(uu.shape), inb.reshape(uu.shape)

    os.makedirs(out_dir, exist_ok=True)
    for f in test[:n_frames]:
        vm = {v: scale[v] * ground_density(view_pts(f, v), grid, sigma) for v in views}
        fused = sum(vm.values()) / len(views)
        gtm = ground_density(ds.gt_points(f)[:, :2], grid, sigma)
        st = {v: stats(vm[v]) for v in views}
        st["fused"] = stats(fused)
        st["ground truth"] = stats(gtm)
        crowded = st["fused"]["p90"] >= 0.72
        chosen = mode if mode != "auto" else ("relative" if crowded else "absolute")
        print(f"\n  frame {f}: fused p90 of occupied area {st['fused']['p90']:.2f} people/m2 (level {los_letter(st['fused']['p90'])}) -> {'crowded' if crowded else 'not crowded'} -> colour scale used for the final picture: {chosen}")
        print("    map            people | occupied area m2 | mean   p90   peak (people/m2) | share of occupied area per level A / B / C / D / E / F")
        for name in views + ["fused", "ground truth"]:
            r = st[name]
            print(f"    {name:13s} {r['count']:7.1f} | {r['occ_area']:8.0f} of {area:.0f}  | {r['mean']:.2f}  {r['p90']:.2f}  {r['peak']:.2f}  | " + " / ".join(f"{x:.0%}" for x in r["shares"]))

        imgs = {}
        try:
            imgs = ds.load_frame_images(f)
        except Exception:
            imgs = {}
        fig, axes = plt.subplots(3, 5, figsize=(23, 12.5))
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        for j, v in enumerate(views):
            ax = axes[0, j]
            if v in imgs:
                im = cv2.resize(imgs[v], (imgs[v].shape[1] // 4, imgs[v].shape[0] // 4))
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                d, valid = overlay_samples(ds.cameras[v], vm[v])
                rgba = colorize(d, chosen, valid, 0.6, ref_of(vm[v]))
                ax.imshow(rgba, extent=(0, im.shape[1], im.shape[0], 0), interpolation="nearest")
            ax.set_title(f"{v} camera view, own heatmap ({chosen} scale)", fontsize=10)
        for row, m in ((1, "absolute"), (2, "relative")):
            for j, (name, d) in enumerate([(v, vm[v]) for v in views] + [("fused (average of 3)", fused), ("ground truth", gtm)]):
                ax = axes[row, j]
                ax.set_facecolor("#1b1b1b")
                ax.imshow(colorize(d, m, mask, 0.95, ref_of(d)), extent=ext, interpolation="nearest")
                ax.set_title(f"ground plane, {m}: {name}\n{st['fused' if 'fused' in name else ('ground truth' if name == 'ground truth' else name)]['count']:.0f} people, p90 {st['fused' if 'fused' in name else ('ground truth' if name == 'ground truth' else name)]['p90']:.2f}/m2", fontsize=9)
        axl = axes[0, 3]
        axl.axis("off")
        if chosen == "absolute":
            sm = cm.ScalarMappable(norm=colors.Normalize(0, abs_max), cmap=cmap_abs)
            cb = fig.colorbar(sm, ax=axl, fraction=0.3, shrink=0.8, pad=0.0)
            ticks = [0.0] + [e for e in LOS_EDGES if e <= abs_max] + [abs_max]
            cb.set_ticks(ticks)
            cb.set_ticklabels(["0"] + [f"{e:.2f}  ({LOS_LETTERS[i]}|{LOS_LETTERS[i + 1]})" for i, e in enumerate(LOS_EDGES) if e <= abs_max] + [f">= {abs_max:.2f}"][:1])
            cb.set_label("people per m2 (absolute)")
        else:
            ref = ref_of(fused)
            handles = []
            for k in range(5):
                lo = ref[int(rel_edges[k - 1] * ref.size)] if k > 0 and ref.size else 0.0
                handles.append(mpatches.Patch(color=rel_cols[k], label=f"{rel_names[k]}" + (f"  (>= {lo:.2f}/m2)" if k > 0 else "")))
            axl.legend(handles=handles, loc="center", fontsize=10, title="relative scale: where it is busier than elsewhere in this frame")
        axes[0, 4].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"heatmap_{f}.png"), dpi=80)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(9, 8))
        ax2.set_facecolor("#1b1b1b")
        ax2.imshow(colorize(fused, chosen, mask, 0.95, ref_of(fused)), extent=ext, interpolation="bilinear")
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("y (m)")
        ax2.set_title(f"frame {f}: fused crowd density, {chosen} scale | {st['fused']['count']:.0f} people | busiest 10%: {st['fused']['p90']:.2f}/m2 (level {los_letter(st['fused']['p90'])}), peak {st['fused']['peak']:.2f}/m2", fontsize=10)
        if chosen == "absolute":
            sm = cm.ScalarMappable(norm=colors.Normalize(0, abs_max), cmap=cmap_abs)
            cb = fig2.colorbar(sm, ax=ax2, fraction=0.046, pad=0.02)
            cb.set_label("people per m2")
            cb.set_ticks([0.0] + [e for e in LOS_EDGES if e <= abs_max] + [abs_max])
        else:
            ref = ref_of(fused)
            hs = [mpatches.Patch(color=rel_cols[k], label=rel_names[k] + (f" (>= {ref[int(rel_edges[k - 1] * ref.size)]:.2f}/m2)" if k > 0 and ref.size else "")) for k in range(5)]
            ax2.legend(handles=hs, loc="lower right", fontsize=8)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, f"heatmap_final_{f}.png"), dpi=100)
        plt.close(fig2)
    print(f"\n  pictures saved to {out_dir}/heatmap_<frame>.png (rows: camera views with their own heatmap; ground plane absolute scale; ground plane relative scale) and heatmap_final_<frame>.png (fused map, chosen scale)")


def epipolar_report(ds, L, frames, tol=8.0):
    hs = np.linspace(1.3, 2.1, 17)
    variants = [("as loaded", dict(ds.cameras))]
    if all(hasattr(c, "dist_model") for c in ds.cameras.values()):
        variants = [(m, {v: replace(c, dist_model=m) for v, c in ds.cameras.items()}) for m in ("tsai", "quartic", "official")]
    print(f"\nEPIPOLAR CONSISTENCY (labels only, no ground truth): for each label in view A, is there a label in view B within {tol:.0f} px of its line of sight (heads 1.3-2.1 m)?")
    print("  distortion model: tsai = kappa1 only (official Image2World) | quartic = kappa1 r^2 + kappa2 r^4 | official = official World2Image and its exact inverse")
    print("  model        A -> B   | partner within 8 px | within 20 px | chance level (labels from another frame, 8 px)")
    for name, cams in variants:
        for a in cams:
            for b in cams:
                if a == b:
                    continue
                hit8 = hit20 = chance = total = 0
                for i, fid in enumerate(frames):
                    la = np.asarray(L[(fid, a)]["pts"], float).reshape(-1, 2)
                    lb = np.asarray(L[(fid, b)]["pts"], float).reshape(-1, 2)
                    lo = np.asarray(L[(frames[(i + 1) % len(frames)], b)]["pts"], float).reshape(-1, 2)
                    if not len(la) or not len(lb):
                        continue
                    samples = []
                    for h in hs:
                        G, ok = cams[a].backproject_ground(la, h)
                        uv, vb = cams[b].project(G, check_bounds=False)
                        uv = np.where((ok & vb)[:, None], uv, 1e9)
                        samples.append(uv)
                    S = np.stack(samples, axis=1)

                    def best(lab):
                        if not len(lab):
                            return np.full(len(la), np.inf)
                        D = np.hypot(S[:, :, None, 0] - lab[None, None, :, 0], S[:, :, None, 1] - lab[None, None, :, 1])
                        return D.min(axis=(1, 2))

                    d = best(lb)
                    hit8 += int((d <= 8).sum())
                    hit20 += int((d <= 20).sum())
                    chance += int((best(lo) <= 8).sum())
                    total += len(la)
                print(f"  {name:11s} {a}->{b} | {hit8 / max(total, 1):18.1%} | {hit20 / max(total, 1):11.1%} | {chance / max(total, 1):.1%}")


def offset_report(ds, L, frames, radius=120.0, bin_px=4.0, near_px=8.0):
    rows_b = [(0, 700), (700, 900), (900, 1520)]
    cols_b = [(0, 900), (900, 1800), (1800, 2704)]
    nb = int(2 * radius / bin_px)
    print(f"\nOFFSET PATTERN: where do the labels sit relative to the projected ground-truth heads? (frames {frames})")
    print("  each cell: dominant offset dx,dy in px (label minus projection) | share of GT heads with a label within 8 px of that offset | share within 8 px with no offset")
    for v, cam in ds.cameras.items():
        recs = []
        for fid in frames:
            gt = ds.gt_points(fid)
            lab = np.asarray(L[(fid, v)]["pts"], float).reshape(-1, 2)
            if not len(gt) or not len(lab):
                continue
            uv, ok = cam.project(gt)
            for p in uv[ok]:
                d = lab - p
                recs.append((p[0], p[1], d[np.hypot(d[:, 0], d[:, 1]) <= radius + near_px]))
        print(f"  {v}")
        print("    image row \\ column " + "".join(f"{f'{a}-{b}':>30s}" for a, b in cols_b))
        for ra, rb in rows_b:
            cells = []
            for ca, cb in cols_b:
                sel = [r for r in recs if ra <= r[1] < rb and ca <= r[0] < cb]
                if len(sel) < 15:
                    cells.append(f"{'(few points)':>30s}")
                    continue
                allv = np.concatenate([r[2] for r in sel]) if any(len(r[2]) for r in sel) else np.zeros((0, 2))
                if not len(allv):
                    cells.append(f"{'(no labels)':>30s}")
                    continue
                H, xe, ye = np.histogram2d(allv[:, 0], allv[:, 1], bins=nb, range=[[-radius, radius], [-radius, radius]])
                Hs = H.copy()
                Hs[1:-1, 1:-1] = sum(H[1 + i:H.shape[0] - 1 + i, 1 + j:H.shape[1] - 1 + j] for i in (-1, 0, 1) for j in (-1, 0, 1))
                i, j = np.unravel_index(np.argmax(Hs), Hs.shape)
                pk = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
                hit_pk = np.mean([bool(len(r[2]) and (np.hypot(r[2][:, 0] - pk[0], r[2][:, 1] - pk[1]) <= near_px).any()) for r in sel])
                hit_0 = np.mean([bool(len(r[2]) and (np.hypot(r[2][:, 0], r[2][:, 1]) <= near_px).any()) for r in sel])
                cells.append(f"{f'{pk[0]:+.0f},{pk[1]:+.0f} | {hit_pk:.0%} | {hit_0:.0%} (n={len(sel)})':>30s}")
            print(f"    {ra:4d}-{rb:<4d}" + " " * 9 + "".join(cells))


def refit_cameras(ds, L, tune, test, frames, views):
    print(f"\nCAMERA REFIT: fitting {', '.join(views)} on tune frames {tune} (ground-truth 3D head points vs image labels), checking on test frames")
    reproj_report(ds, L, test)
    saved = {}
    for v in views:
        cam = ds.cameras[v]
        world = [ds.gt_points(f) for f in tune]
        labels = [L[(f, v)]["pts"] for f in tune]
        new, info = refine_camera(cam, world, labels)
        th = info["params"]
        print(f"  {v}: {info['pairs']} pairs used | rotation {np.degrees(th[:3]).round(2)} deg | shift {th[3:6].round(0)} mm | focal x{np.exp(th[6]):.4f} | kappa {th[7]:+.4f}")
        ds.cameras[v] = new
        saved[v] = dict(R=new.R.tolist(), t=new.t.tolist(), focal=float(new.focal), kappa1=float(new.kappa1), cx=float(new.cx), cy=float(new.cy))
    L = build_labels(ds, frames)
    print("AFTER REFIT:")
    reproj_report(ds, L, test)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "camera_refit.json"), "w") as f:
        json.dump(saved, f, indent=1)
    print(f"  refined camera parameters saved to {OUT_DIR}/camera_refit.json")
    return L


def fmt_pct(x):
    return "  n/a" if x != x else f"{100 * x:5.1f}%"


def print_sweep(title, rows, mark=None):
    print(f"\n{title}\n conf   boxes  labels    TP    FP    FN   Prec   Rec    F1")
    for r in rows:
        star = "  <-- chosen" if mark is not None and abs(r["conf"] - mark) < 1e-9 else ""
        print(f" {r['conf']:4.2f}  {r['boxes']:6d} {r['labels']:7d} {r['tp']:5d} {r['fp']:5d} {r['fn']:5d}  "
              f"{r['p']:.3f} {r['r']:.3f} {r['f1']:.3f}{star}")


def save_error_overlays(ds, dets, L, frames, thr, out_dir, limit):
    os.makedirs(out_dir, exist_ok=True)
    for fid in frames[:limit]:
        imgs = ds.load_frame_images(fid)
        for v, img in imgs.items():
            k = (fid, v)
            _, _, _, rows = image_metrics(dets, L, [k], thr, detail=True)
            _, bt, m, hit = rows[0]
            matched_b = {i for i, _ in m}
            out = img.copy()
            for i, (x0, y0, x1, y1) in enumerate(bt):
                col = (0, 200, 0) if i in matched_b else (0, 0, 255)
                cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), col, 3)
            for j, (x, y) in enumerate(L[k]["pts"]):
                if not hit[j]:
                    cv2.circle(out, (int(x), int(y)), 16, (0, 255, 255), 4)
            cv2.putText(out, f"green=TP red=FP yellow circle=missed label | {v} frame {fid} conf>={thr}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
            cv2.imwrite(os.path.join(out_dir, f"errors_{fid}_{v}.jpg"), cv2.resize(out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))


def plot_summary(path, tune_sw, test_sw, thr, size_tbl, gt_counts, view_tbl, dist_tbl, split):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    a = ax[0, 0]
    for sw, name, ls in ((tune_sw, "tune", "--"), (test_sw, "test", "-")):
        if sw:
            a.plot([r["r"] for r in sw], [r["p"] for r in sw], ls, marker="o", ms=3, label=f"{name} (AP~{average_precision(sw):.2f})")
    a.set_xlabel("recall"); a.set_ylabel("precision"); a.set_title("Precision-recall (image space)"); allr = [r["r"] for sw in (tune_sw, test_sw) for r in sw] or [0.0]
    allp = [r["p"] for sw in (tune_sw, test_sw) for r in sw] or [0.0]
    a.set_xlim(max(0.0, min(allr) - 0.08), 1.01); a.set_ylim(max(0.0, min(allp) - 0.08), 1.01); a.legend(); a.grid(alpha=0.3)
    a = ax[0, 1]
    sw = test_sw or tune_sw
    for key, lab in (("p", "precision"), ("r", "recall"), ("f1", "F1")):
        a.plot([r["conf"] for r in sw], [r[key] for r in sw], marker="o", ms=3, label=lab)
    a.axvline(thr, color="k", ls=":", label=f"chosen {thr}")
    a.set_xlabel("confidence threshold"); a.set_title("Metrics vs confidence"); a.legend(); a.grid(alpha=0.3)
    a = ax[0, 2]
    labels = [f"{lo:.0f}-{hi:.0f}" if hi < 1e8 else f">{lo:.0f}" for lo, hi, _, _ in size_tbl]
    a.bar(labels, [0 if r != r else r for _, _, _, r in size_tbl], color="#3b7dd8")
    for i, (_, _, n, r) in enumerate(size_tbl):
        a.text(i, (0 if r != r else r) + 0.02, f"n={n}", ha="center", fontsize=8)
    a.set_ylim(0, 1.12); a.set_xlabel("expected person height in image (px)"); a.set_ylabel("recall"); a.set_title(f"Recall by apparent size ({split})")
    a = ax[1, 0]
    for mv, c in ((1, "#d83b3b"), (2, "#3b9d3b")):
        if mv in gt_counts and gt_counts[mv][0]:
            g, p = np.array(gt_counts[mv][0]), np.array(gt_counts[mv][1])
            a.scatter(g, p, c=c, label=f"min_views={mv}", alpha=0.8)
    allv = [x for mv in gt_counts for x in gt_counts[mv][0] + gt_counts[mv][1]]
    if allv:
        lim = [min(allv) - 2, max(allv) + 2]
        a.plot(lim, lim, "k--", lw=1)
    a.set_xlabel("ground-truth count"); a.set_ylabel("predicted count"); a.set_title("Count per frame (ground plane)"); a.legend(); a.grid(alpha=0.3)
    a = ax[1, 1]
    a.bar([str(k) for k, _, _ in view_tbl], [0 if r != r else r for _, _, r in view_tbl], color="#8a4fd8")
    for i, (_, n, r) in enumerate(view_tbl):
        a.text(i, (0 if r != r else r) + 0.02, f"n={n}", ha="center", fontsize=8)
    a.set_ylim(0, 1.12); a.set_xlabel("cameras that can see the person"); a.set_ylabel("ground-plane recall"); a.set_title(f"Recall vs camera coverage ({split})")
    a = ax[1, 2]
    dl = [f"{lo:.0f}-{hi:.0f}" if hi < 1e8 else f">{lo:.0f}" for lo, hi, _, _ in dist_tbl]
    a.bar(dl, [0 if r != r else r for _, _, _, r in dist_tbl], color="#d8943b")
    for i, (_, _, n, r) in enumerate(dist_tbl):
        a.text(i, (0 if r != r else r) + 0.02, f"n={n}", ha="center", fontsize=8)
    a.set_ylim(0, 1.12); a.set_xlabel("distance from camera (m)"); a.set_ylabel("recall"); a.set_title(f"Recall by distance ({split})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def compare_strategies(ds, base, frames, L, use_cache):
    keys = [(f, v) for f in frames for v in ds.cameras]
    rows = []
    print("\n" + "=" * 78 + "\nSTRATEGY COMPARISON (all selected frames, best-F1 threshold per strategy)\n" + "=" * 78)
    for name, over in STRATEGIES.items():
        cfg = YoloConfig(**{**base, **over, "conf": SWEEP_MIN})
        dets, sec = detect_all(ds, cfg, frames, use_cache)
        sw = sweep(dets, L, keys)
        best = max(sw, key=lambda r: r["f1"])
        H, _, hit, _, _ = collect_label_hits(dets, L, keys, best["conf"])
        small = hit[np.isfinite(H) & (H < 60)]
        rows.append((name, best["conf"], best["p"], best["r"], best["f1"], float(small.mean()) if len(small) else float("nan"), sec))
    print(f"{'strategy':30s} {'conf':>5s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'recall<60px':>12s} {'s/view':>13s}")
    for n, c, p, r, f1, sm, sec in rows:
        print(f"{n:30s} {c:5.2f} {p:6.3f} {r:6.3f} {f1:6.3f} {fmt_pct(sm):>12s} {('n/a (cached)' if sec is None else f'{sec:5.1f}'):>13s}")
    write_csv(os.path.join(OUT_DIR, "strategy_comparison.csv"), ["strategy", "conf", "precision", "recall", "f1", "recall_under_60px", "sec_per_view"], rows)


def main():
    global ASSOC_TOL
    ap = argparse.ArgumentParser(description="Evaluate YOLO person detection on CityStreet")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-tune", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--tune-frames", default="")
    ap.add_argument("--test-frames", default="")
    ap.add_argument("--conf", type=float, default=None, help="force the final threshold instead of tuning it")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--save-errors", type=int, default=2)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--reproj", action="store_true")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--offsets", action="store_true")
    ap.add_argument("--epipolar", action="store_true")
    ap.add_argument("--density", action="store_true")
    ap.add_argument("--heatmaps", action="store_true")
    ap.add_argument("--heatmap-mode", choices=["auto", "absolute", "relative"], default="auto")
    ap.add_argument("--heat-point", choices=["head", "mid", "feet"], default="mid")
    ap.add_argument("--abs-max", type=float, default=1.0)
    ap.add_argument("--heat-frames", type=int, default=2)
    ap.add_argument("--dist-model", choices=["tsai", "quartic", "official"], default=None)
    ap.add_argument("--refit-views", default="view2,view3")
    ap.add_argument("--assoc-tol", type=float, default=ASSOC_TOL)
    args = ap.parse_args()
    ASSOC_TOL = args.assoc_tol
    if (args.density or args.heatmaps) and not args.dist_model:
        args.dist_model = "tsai"

    base = dict(YOLO)
    if args.model:
        base["model"] = args.model
    if args.device:
        base["device"] = args.device
    ds = CityStreetData(args.data_root)
    if args.dist_model:
        ds.cameras = {v: replace(c, dist_model=args.dist_model) for v, c in ds.cameras.items()}
        print(f"distortion model for all cameras: {args.dist_model}")
    if not len(ds.gt()):
        sys.exit(f"No ground truth found under {args.data_root} (expected {ds.gt_h5})")
    tune, test, usable = pick_frames(ds, args)
    if not (tune or test):
        sys.exit(f"No frames with GT + VIA labels in all views were found ({len(usable)} usable).")
    frames = sorted(set(tune) | set(test))
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"model {base['model']} | tune frames {tune} | test frames {test}")
    print(f"est. detection time without cache: {len(frames) * 3} views (about 40 s/view on CPU, about 1 s/view on GPU)")

    cfg = YoloConfig(**{**base, "conf": SWEEP_MIN})
    dets, _ = detect_all(ds, cfg, frames, not args.no_cache)
    L = build_labels(ds, frames)
    if args.refit:
        L = refit_cameras(ds, L, tune, test or tune, frames, [x.strip() for x in args.refit_views.split(",") if x.strip()])
    if args.oracle:
        oracle_report(ds, L, test or tune)
    if args.reproj:
        reproj_report(ds, L, test or tune)
    if args.offsets:
        offset_report(ds, L, test or tune)
    if args.epipolar:
        epipolar_report(ds, L, test or tune)
    keys_t = [(f, v) for f in tune for v in ds.cameras]
    keys_e = [(f, v) for f in test for v in ds.cameras]
    keys_all = keys_t + keys_e

    n_labels = sum(len(L[k]["pts"]) for k in keys_all)
    out_roi = sum(int((~L[k]["inroi"]).sum()) for k in keys_all)
    print(f"\nlabels: {n_labels} across {len(keys_all)} views | outside camera ROI (unreachable by the ROI filter): {out_roi}")

    tune_sw = sweep(dets, L, keys_t) if keys_t else []
    test_sw = sweep(dets, L, keys_e) if keys_e else []
    if args.conf is not None:
        thr, how = args.conf, "forced by --conf"
    elif tune_sw:
        thr, how = max(tune_sw, key=lambda r: r["f1"])["conf"], "best F1 on tune frames"
    else:
        thr, how = max(test_sw, key=lambda r: r["f1"])["conf"], "best F1 on TEST frames (no tune frames: optimistic)"
    print(f"chosen confidence threshold: {thr} ({how})")
    if args.density:
        density_report(ds, dets, tune, test or tune, thr, OUT_DIR)
        return
    if args.heatmaps:
        heatmap_report(ds, dets, tune, test or tune, thr, OUT_DIR, args.heatmap_mode, args.heat_point, args.abs_max, args.heat_frames)
        return
    if tune_sw:
        print_sweep("IMAGE-SPACE SWEEP: tune frames", tune_sw, thr)
    if test_sw:
        print_sweep("IMAGE-SPACE SWEEP: test frames (held out)", test_sw, thr)
    rep_keys = keys_e or keys_t
    rep_sw = test_sw or tune_sw
    at = min(rep_sw, key=lambda r: abs(r["conf"] - thr))
    print(f"\nHEADLINE (image space, {'test' if keys_e else 'tune'} frames, conf {thr}): "
          f"P {at['p']:.3f}  R {at['r']:.3f}  F1 {at['f1']:.3f}  AP~{average_precision(rep_sw):.3f}")

    H, D, hit, view, inroi = collect_label_hits(dets, L, rep_keys, thr)
    _, _, _, rows = image_metrics(dets, L, rep_keys, thr, detail=True)
    rel = []
    for k, bt, m, _ in rows:
        for i, j in m:
            rel.append((L[k]["pts"][j, 1] - bt[i, 1]) / max(bt[i, 3] - bt[i, 1], 1.0))
    rel_med = float(np.median(rel)) if rel else float("nan")
    print(f"label position inside matched boxes (0=top, 1=bottom): {rel_med:.2f}"
          f"  ({'looks like HEAD labels' if rel_med < 0.3 else 'looks like body-centre labels: size/ground estimates are biased'})")

    near, missed = near_miss_stats(rows, L)
    if missed:
        print(f"missed labels: {missed} | of these {near} ({fmt_pct(near / missed).strip()}) have an UNMATCHED box right next to them"
              " -> the person was found but the label fell outside the box (label/box alignment), not a true miss")

    size_tbl = recall_by(H, hit, SIZE_BINS)
    print("\nRECALL BY EXPECTED PERSON HEIGHT IN IMAGE")
    for lo, hi, n, r in size_tbl:
        print(f"  {lo:4.0f}-{hi if hi < 1e8 else 'inf':>4} px : recall {fmt_pct(r)}  (n={n})")
    dist_edges = [0, 15, 25, 35, 50, 1e9]
    dist_tbl = recall_by(D, hit, dist_edges)
    print("RECALL BY DISTANCE FROM CAMERA")
    for lo, hi, n, r in dist_tbl:
        print(f"  {lo:4.0f}-{hi if hi < 1e8 else 'inf':>4} m  : recall {fmt_pct(r)}  (n={n})")
    print("RECALL BY CAMERA")
    for v in ds.cameras:
        m = view == v
        print(f"  {v}: recall {fmt_pct(float(hit[m].mean()) if m.any() else float('nan'))}  (n={int(m.sum())})")
    write_csv(os.path.join(OUT_DIR, "size_recall.csv"), ["h_lo_px", "h_hi_px", "n_labels", "recall"], size_tbl)
    write_csv(os.path.join(OUT_DIR, "conf_sweep.csv"), ["split", "conf", "boxes", "labels", "tp", "fp", "fn", "precision", "recall", "f1"],
              [[s, r["conf"], r["boxes"], r["labels"], r["tp"], r["fp"], r["fn"], r["p"], r["r"], r["f1"]]
               for s, sw in (("tune", tune_sw), ("test", test_sw)) for r in sw])

    gframes = test or tune
    print(f"\nGROUND-PLANE EVALUATION ({'test' if test else 'tune'} frames {gframes}, conf {thr}, "
          f"fuse {FUSE_RADIUS} m, match {MATCH_RADIUS} m)")
    reg = gt_region(ds)
    print(f"  predictions clipped to GT region x [{reg[0]:.1f}, {reg[1]:.1f}] y [{reg[2]:.1f}, {reg[3]:.1f}] (head/noclip row is unclipped); triang = height-aware association, tol {ASSOC_TOL} m")
    print("  projection    min_views |   TP   FP   FN   Prec   Rec    F1 | count MAE  RMSE  bias  rel.err | median loc err")
    store, gt_counts, view_tbl, tri_tbl = {}, {}, [], []
    variants = [("head", "head", True), ("head/noclip", "head", False), ("feet", "feet", True), ("triang", "tri", True)]
    for label, mode, clip in variants:
        res, gt_vis = ground_eval(ds, dets, gframes, thr, mode, clip=clip)
        store[label] = (res, gt_vis)
        for mv, r in res.items():
            p, rc, f1 = prf(r["tp"], r["fp"], r["fn"])
            e = np.array(r["err"], float)
            rel_e = float(np.mean(np.abs(e) / np.maximum(np.array(r["gt"], float), 1))) if len(e) else float("nan")
            print(f"  {label:12s}  {mv:9d} | {r['tp']:4d} {r['fp']:4d} {r['fn']:4d}  {p:.3f} {rc:.3f} {f1:.3f} | "
                  f"{np.mean(np.abs(e)):9.1f} {np.sqrt(np.mean(e ** 2)):5.1f} {np.mean(e):+5.1f}  {fmt_pct(rel_e)} | "
                  f"{(np.median(r['d']) if r['d'] else float('nan')):.2f} m")
        if label in ("head", "triang"):
            vis = np.array(gt_vis) if gt_vis else np.zeros((0, 2), int)
            tbl = []
            for k in range(0, 4):
                m = vis[:, 0] == k if len(vis) else np.zeros(0, bool)
                tbl.append((k, int(m.sum()), float(vis[m, 1].mean()) if m.any() else float("nan")))
            if label == "head":
                gt_counts = {mv: (r["gt"], r["pred"]) for mv, r in res.items()}
                view_tbl = tbl
            else:
                tri_tbl = tbl
    print("GROUND-PLANE RECALL vs NUMBER OF CAMERAS THAT CAN SEE THE PERSON (min_views=1)")
    for (k, n, r), (_, _, r2) in zip(view_tbl, tri_tbl):
        print(f"  {k} camera(s): head {fmt_pct(r)} | triang {fmt_pct(r2)}  (n={n})")

    pf = []
    for fid in frames:
        row = [fid, "tune" if fid in tune else "test", len(ds.gt_points(fid))]
        for v in ds.cameras:
            k = (fid, v)
            tp, fp, fn = image_metrics(dets, L, [k], thr)
            row += [int((dets[k][1] >= thr).sum()), tp, fp, fn]
        r1, _ = ground_eval(ds, dets, [fid], thr, "head")
        row += [r1[1]["pred"][0] if r1[1]["pred"] else 0, r1[2]["pred"][0] if r1[2]["pred"] else 0]
        pf.append(row)
    hdr = ["frame", "split", "gt_count"] + [f"{v}_{x}" for v in ds.cameras for x in ("boxes", "tp", "fp", "fn")] + ["pred_minviews1", "pred_minviews2"]
    write_csv(os.path.join(OUT_DIR, "per_frame.csv"), hdr, pf)

    plot_summary(os.path.join(OUT_DIR, "summary.png"), tune_sw, test_sw, thr, size_tbl, gt_counts, view_tbl, dist_tbl, "test" if test else "tune")
    if args.save_errors > 0:
        save_error_overlays(ds, dets, L, gframes, thr, os.path.join(OUT_DIR, "errors"), args.save_errors)

    weak = [(lo, hi, r) for lo, hi, n, r in size_tbl if n >= 10 and r == r]
    print("\nREADING THE RESULTS")
    if weak:
        lo, hi, r = min(weak, key=lambda x: x[2])
        print(f"  weakest size bucket: {lo:.0f}-{hi if hi < 1e8 else 'inf'} px with recall {fmt_pct(r)}"
              + ("  -> small people limit accuracy: try --tile 640 --tile-imgsz 960 style settings or a larger model" if r < 0.7 else ""))
    print(f"  count bias at min_views=1: {np.mean(store['head'][0][1]['err']):+.1f} people/frame"
          " (positive = over-count: duplicates/false positives; negative = missed people)")
    print(f"  outputs written to {OUT_DIR}/  (summary.png, conf_sweep.csv, size_recall.csv, per_frame.csv, errors/)")

    if args.compare:
        compare_strategies(ds, base, frames, L, not args.no_cache)


if __name__ == "__main__":
    main()