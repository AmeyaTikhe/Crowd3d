#!/usr/bin/env python3
"""crowd3d_all.py: YOLO person detection, ground-plane projection and multi-view fusion for CityStreet.

Library used by ../run_images.py. Benchmark on the dataset (confidence sweep + ground-plane metrics):

    python crowd3d/crowd3d_all.py eval --data-root CityStreet --frames 636,700,800
    python crowd3d/crowd3d_all.py eval --data-root CityStreet --model yolo26x.pt --device 0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial.transform import Rotation


# --- 1. CONFIG ---
@dataclass
class YoloConfig:
    model: str = "yolo11x.pt"
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


@dataclass
class GroundGrid:
    x_min: float = -18.0
    x_max: float = 25.0
    y_min: float = -28.0
    y_max: float = 24.0
    cell: float = 0.25

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(round((self.y_max - self.y_min) / self.cell)), int(round((self.x_max - self.x_min) / self.cell)))

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        return (self.x_min, self.x_max, self.y_min, self.y_max)


# --- 2. GEOMETRY ---
CITYSTREET_RES_SCALER_M = 0.07625
CITYSTREET_BBOX_X = 352.0 * 0.8
CITYSTREET_BBOX_Y = 522.0 * 0.8
VIEW_TO_CAMERA_DIR = {"view1": "camera1", "view2": "camera3", "view3": "camera4"}


def world_m_to_groundplane_px(X_m, Y_m):
    return X_m / CITYSTREET_RES_SCALER_M + CITYSTREET_BBOX_X, Y_m / CITYSTREET_RES_SCALER_M + CITYSTREET_BBOX_Y


def groundplane_px_to_world_m(cx, cy):
    return (cx - CITYSTREET_BBOX_X) * CITYSTREET_RES_SCALER_M, (cy - CITYSTREET_BBOX_Y) * CITYSTREET_RES_SCALER_M


@dataclass
class Camera:
    view_id: str
    camera_name: str
    img_w: int = 2704
    img_h: int = 1520
    cx: float = 0.0
    cy: float = 0.0
    dpx: float = 0.0
    dpy: float = 0.0
    focal: float = 1.0
    sx: float = 1.0
    kappa1: float = 0.0
    kappa2: float = 0.0
    R: np.ndarray = None
    t: np.ndarray = None
    center_m: np.ndarray = None
    dist_model: str = "quartic"

    def __post_init__(self):
        if self.R is not None and self.t is not None:
            C = -self.R.T @ self.t
            self.center_m = np.array([C[0] / 1000.0, C[1] / 1000.0, -C[2] / 1000.0], dtype=np.float64)

    def _invert_distortion(self, Ru: np.ndarray):
        k1 = self.kappa1
        k2 = 0.0 if self.dist_model == "tsai" else self.kappa2
        Rd = Ru.copy()
        for _ in range(12):
            Rd2 = Rd * Rd
            f = Rd * (1.0 + k1 * Rd2 + k2 * Rd2 * Rd2) - Ru
            d = 1.0 + 3.0 * k1 * Rd2 + 5.0 * k2 * Rd2 * Rd2
            ok_d = np.abs(d) > 1e-6
            Rd = Rd - np.where(ok_d, f / np.where(ok_d, d, 1.0), 0.0)
        resid = np.abs(Rd * (1.0 + k1 * Rd * Rd + k2 * Rd * Rd * Rd * Rd) - Ru)
        return Rd, np.isfinite(Rd) & (resid < 1e-4) & (1.0 + 3.0 * k1 * Rd * Rd + 5.0 * k2 * Rd * Rd * Rd * Rd > 1e-3)

    def _official_forward(self, Xu: np.ndarray, Yu: np.ndarray):
        k1, k2 = self.kappa1, self.kappa2
        a, b, c3, d = k2 * k2, 2.0 * k1 * k2, k1 * k1 + 2.0 * k2, 2.0 * k1
        Ru = np.sqrt(Xu * Xu + Yu * Yu)
        Rd = Ru.copy()
        for _ in range(60):
            f = a * Rd ** 5 + b * Rd ** 4 + c3 * Rd ** 3 + d * Rd ** 2 + Rd - Ru
            df = 5 * a * Rd ** 4 + 4 * b * Rd ** 3 + 3 * c3 * Rd ** 2 + 2 * d * Rd + 1.0
            ok_d = np.abs(df) > 1e-9
            Rd = Rd - np.where(ok_d, f / np.where(ok_d, df, 1.0), 0.0)
        f = a * Rd ** 5 + b * Rd ** 4 + c3 * Rd ** 3 + d * Rd ** 2 + Rd - Ru
        lam = 1.0 + k1 * Rd + k2 * Rd * Rd
        good = np.isfinite(Rd) & (np.abs(f) < 1e-8) & (np.abs(lam) > 1e-6)
        lam = np.where(good, lam, 1.0)
        return Xu / lam, Yu / lam, good

    def _official_inverse_factor(self, r: np.ndarray):
        k1, k2 = self.kappa1, self.kappa2
        Rd = r.copy()
        for _ in range(60):
            f = k2 * Rd ** 3 + k1 * Rd ** 2 + Rd - r
            df = 3 * k2 * Rd ** 2 + 2 * k1 * Rd + 1.0
            ok_d = np.abs(df) > 1e-9
            Rd = Rd - np.where(ok_d, f / np.where(ok_d, df, 1.0), 0.0)
        return 1.0 + k1 * Rd + k2 * Rd * Rd

    def project(self, points_3d_m: np.ndarray, check_bounds: bool = True):
        pts = np.asarray(points_3d_m, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            return np.zeros((0, 2)), np.zeros((0,), dtype=bool)
        P_world = np.stack([pts[:, 0] * 1000.0, pts[:, 1] * 1000.0, -pts[:, 2] * 1000.0], axis=1)
        P_cam = (self.R @ P_world.T).T + self.t
        xc, yc, zc = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            Xu, Yu = self.focal * xc / zc, self.focal * yc / zc
            Ru = np.sqrt(Xu * Xu + Yu * Yu)
            Ru_safe = np.where(np.isfinite(Ru), Ru, 0.0)
            if self.dist_model == "official":
                Xs, Ys = np.where(np.isfinite(Xu), Xu, 0.0), np.where(np.isfinite(Yu), Yu, 0.0)
                Xdd, Ydd, dist_ok = self._official_forward(Xs, Ys)
                u = (Xdd * self.sx) / self.dpx + self.cx
                v = Ydd / self.dpy + self.cy
            else:
                Rd, dist_ok = self._invert_distortion(Ru_safe)
                scale = np.where(Ru_safe > 1e-12, Rd / np.where(Ru_safe > 1e-12, Ru_safe, 1.0), 1.0)
                u = (Xu * scale * self.sx) / self.dpx + self.cx
                v = (Yu * scale) / self.dpy + self.cy
        uv = np.stack([u, v], axis=1)
        valid = (zc > 1.0) & dist_ok & np.isfinite(uv).all(axis=1)
        if check_bounds:
            valid &= (u >= 0.0) & (u < float(self.img_w)) & (v >= 0.0) & (v < float(self.img_h))
        return uv, valid

    def backproject_ground(self, pixels_uv: np.ndarray, plane_z_m: float = 0.0):
        uv = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 2)
        N = uv.shape[0]
        if N == 0:
            return np.zeros((0, 3)), np.zeros((0,), dtype=bool)
        Xd = self.dpx * (uv[:, 0] - self.cx) / self.sx
        Yd = self.dpy * (uv[:, 1] - self.cy)
        R2 = Xd * Xd + Yd * Yd
        if self.dist_model == "official":
            fac = self._official_inverse_factor(np.sqrt(R2))
        elif self.dist_model == "tsai":
            fac = 1.0 + self.kappa1 * R2
        else:
            fac = 1.0 + self.kappa1 * R2 + self.kappa2 * R2 * R2
        Xu, Yu = Xd * fac, Yd * fac
        Zi = -plane_z_m * 1000.0
        R, t = self.R, self.t
        A00 = R[0, 0] - Xu * R[2, 0]
        A01 = R[0, 1] - Xu * R[2, 1]
        b0 = (Xu * R[2, 2] - R[0, 2]) * Zi + (Xu * t[2] - t[0])
        A10 = R[1, 0] - Yu * R[2, 0]
        A11 = R[1, 1] - Yu * R[2, 1]
        b1 = (Yu * R[2, 2] - R[1, 2]) * Zi + (Yu * t[2] - t[1])
        det = A00 * A11 - A01 * A10
        valid = np.abs(det) > 1e-9
        sd = np.where(valid, det, 1.0)
        Xw = np.where(valid, (b0 * A11 - b1 * A01) / sd, 0.0)
        Yw = np.where(valid, (A00 * b1 - A10 * b0) / sd, 0.0)
        return np.stack([Xw / 1000.0, Yw / 1000.0, np.full(N, plane_z_m)], axis=1), valid


def create_citystreet_cameras() -> Dict[str, Camera]:
    cams = {}
    R1 = np.array([[-0.03615359, 0.99596323, -0.0821594], [-0.34251474, 0.06488427, 0.93726927],
                   [0.93881658, 0.06202646, 0.33878628]])
    t1 = np.array([1275.94012233, 226.52473842, 24230.24027107])
    cams["view1"] = Camera("view1", "camera1", 2704, 1520, 1384.55, 747.846499320945, 1.0 / 1243.4, 1.0 / 1248.9,
                           1.0, 1.0, -0.2090, 0.0382, R1, t1)
    R2 = np.array([[-0.99730688, 0.06324747, 0.03713147], [0.02439897, -0.19132768, 0.9812229],
                   [0.06916415, 0.97948633, 0.18926923]])
    t2 = np.array([9680.47673912, 2273.39720387, 31436.76306171])
    cams["view2"] = Camera("view2", "camera3", 2704, 1520, 1292.5376, 770.518094, 1.0 / 1258.18922,
                           1.0 / 1259.88567, 1.0, 1.0, -0.23977946, 0.05901853, R2, t2)
    R3 = np.array([[0.99545502, 0.07578782, -0.05766726], [0.0167508, 0.45675628, 0.88943415],
                   [0.09374816, -0.88635766, 0.45341083]])
    t3 = np.array([3309.55016035, -4603.90350916, 29264.19379322])
    cams["view3"] = Camera("view3", "camera4", 2704, 1520, 1374.1, 785.7148, 1.0 / 1255.9, 1.0 / 1257.2,
                           1.0, 1.0, -0.2489, 0.07147814, R3, t3)
    return cams


def refine_camera(cam: Camera, world_list, label_list, gates=(80.0, 50.0, 30.0, 15.0), prior_px: float = 3.0):
    sig = np.array([0.03, 0.03, 0.03, 500.0, 500.0, 500.0, 0.05, 0.05])
    xs = np.array([0.01, 0.01, 0.01, 100.0, 100.0, 100.0, 0.01, 0.01])
    th = np.zeros(8)

    def adjust(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix() @ cam.R
        return replace(cam, R=R, t=cam.t + x[3:6], focal=cam.focal * float(np.exp(x[6])), kappa1=cam.kappa1 + x[7])

    pairs = 0
    for gate in gates:
        cur = adjust(th)
        W, U = [], []
        for w, lab in zip(world_list, label_list):
            w = np.asarray(w, float).reshape(-1, 3)
            lab = np.asarray(lab, float).reshape(-1, 2)
            if not len(w) or not len(lab):
                continue
            uv, ok = cur.project(w, check_bounds=False)
            idx = np.where(ok & np.isfinite(uv).all(axis=1))[0]
            if not len(idx):
                continue
            D = np.hypot(uv[idx, None, 0] - lab[None, :, 0], uv[idx, None, 1] - lab[None, :, 1])
            a, b = linear_sum_assignment(D)
            g = D[a, b] <= gate
            W.append(w[idx[a[g]]])
            U.append(lab[b[g]])
        if not W:
            break
        W, U = np.concatenate(W), np.concatenate(U)
        if len(W) < 20:
            break

        def resid(x):
            uv, _ = adjust(x).project(W, check_bounds=False)
            r = np.nan_to_num(uv - U, nan=1e3, posinf=1e3, neginf=-1e3)
            return np.concatenate([r.ravel(), prior_px * x / sig])

        th = least_squares(resid, th, x_scale=xs, loss="soft_l1", f_scale=5.0).x
        pairs = len(W)
    return adjust(th), {"pairs": pairs, "params": th}


def build_camera(view_id: str, calib, img_w: int, img_h: int) -> Optional[Camera]:
    """calib: 'citystreet_view1/2/3' | dict with R, t, cx, cy, dpx, dpy, kappa1 ... | None or {} (uncalibrated)."""
    if isinstance(calib, str) and calib.startswith("citystreet_"):
        cam = create_citystreet_cameras().get(calib.replace("citystreet_", ""))
        if cam is None:
            raise ValueError(f"unknown calibration preset: {calib}")
        cam.view_id = view_id
        return cam
    if isinstance(calib, dict) and calib:
        return Camera(view_id=view_id, camera_name=calib.get("camera_name", view_id),
                      img_w=calib.get("img_w", img_w), img_h=calib.get("img_h", img_h),
                      cx=calib.get("cx", img_w / 2.0), cy=calib.get("cy", img_h / 2.0),
                      dpx=calib.get("dpx", 1.0 / (img_w * 0.5)), dpy=calib.get("dpy", 1.0 / (img_h * 0.5)),
                      focal=calib.get("focal", 1.0), sx=calib.get("sx", 1.0), kappa1=calib.get("kappa1", 0.0),
                      R=np.array(calib["R"], np.float64), t=np.array(calib["t"], np.float64))
    return None


def load_roi_mask(roi_dir: str, view_index: int) -> Optional[np.ndarray]:
    p = os.path.join(roi_dir, f"mask{view_index}_ic.npz")
    if not os.path.exists(p):
        return None
    with np.load(p) as d:
        key = "arr_0" if "arr_0" in d.files else d.files[0]
        return (np.asarray(d[key]) > 0).astype(np.uint8)


def load_via_points(path: str) -> Dict[int, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict) and "_via_img_metadata" in d:
        d = d["_via_img_metadata"]
    out: Dict[int, np.ndarray] = {}
    for key, ent in d.items():
        if not isinstance(ent, dict):
            continue
        m = re.search(r"frame_(\d+)", str(ent.get("filename", key)))
        if not m:
            continue
        regs = ent.get("regions", {})
        it = regs.values() if isinstance(regs, dict) else regs
        pts = [(r["shape_attributes"]["cx"], r["shape_attributes"]["cy"]) for r in it
               if "shape_attributes" in r and "cx" in r["shape_attributes"] and "cy" in r["shape_attributes"]]
        out[int(m.group(1))] = np.array(pts, dtype=np.float64).reshape(-1, 2)
    return out


def load_gt_groundplane(h5_path: str) -> np.ndarray:
    """(K,5) rows [frame_id, person_id, cx, cy, height_mm]."""
    with h5py.File(h5_path, "r") as f:
        key = "v_pmap_GP" if "v_pmap_GP" in f else list(f.keys())[0]
        arr = np.asarray(f[key][:], dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] != 5 and arr.shape[0] == 5:
        arr = arr.T
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"unexpected GT array shape {arr.shape}")
    return arr


def gt_points_for_frame(gt: np.ndarray, frame_id: int) -> np.ndarray:
    """(K,3) [X_m, Y_m, height_m]."""
    rows = gt[gt[:, 0] == float(frame_id)]
    if len(rows) == 0:
        return np.zeros((0, 3))
    X, Y = groundplane_px_to_world_m(rows[:, 2], rows[:, 3])
    return np.stack([X, Y, rows[:, 4] / 1000.0], axis=1)


# --- 3. DETECTION ---
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
        from ultralytics import YOLO
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


# --- 4. GROUND PLANE ---
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


H_MIN = 1.3
H_MAX = 2.1


def pixel_rays(cam: Camera, uv: np.ndarray):
    p0, ok0 = cam.backproject_ground(uv, 0.0)
    p2, ok2 = cam.backproject_ground(uv, 2.0)
    d = p2 - p0
    n = np.linalg.norm(d, axis=1)
    ok = ok0 & ok2 & (n > 1e-9)
    d = d / np.where(n > 1e-9, n, 1.0)[:, None]
    return p0, d, ok


def ray_stats(o: np.ndarray, d: np.ndarray):
    A = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    b = np.einsum("nij,nj->ni", A, o)
    c = np.einsum("ni,ni->n", o, b)
    return A, b, c


def _pair_costs(A, b, c, n, Aj, bj, cj):
    As = A[:, None] + Aj[None] + 1e-9 * np.eye(3)
    bs = b[:, None] + bj[None]
    with np.errstate(all="ignore"):
        p = np.linalg.solve(As, bs[..., None])[..., 0]
        s = (c[:, None] + cj[None]) - np.einsum("kmi,kmi->km", bs, p)
        cost = np.maximum(s, 0.0) / (n[:, None] + 1.0)
    return cost, p


def fuse_pixels(cams: Dict[str, Camera], uv_per_view: Dict[str, np.ndarray], score_per_view=None,
                tol: float = 0.4, hmin: float = H_MIN, hmax: float = H_MAX, single_h: float = 1.7,
                bounds=(-30.0, 35.0, -40.0, 35.0)):
    A = np.zeros((0, 3, 3))
    b = np.zeros((0, 3))
    c = np.zeros(0)
    n = np.zeros(0)
    ssum = np.zeros(0)
    sxy = np.zeros((0, 2))
    for v, cam in cams.items():
        uv = np.asarray(uv_per_view.get(v, np.zeros((0, 2))), float).reshape(-1, 2)
        if len(uv) == 0:
            continue
        sc = np.ones(len(uv)) if score_per_view is None else np.asarray(score_per_view[v], float)
        o, d, ok = pixel_rays(cam, uv)
        uv, sc, o, d = uv[ok], sc[ok], o[ok], d[ok]
        if len(uv) == 0:
            continue
        Aj, bj, cj = ray_stats(o, d)
        xy1, _ = cam.backproject_ground(uv, single_h)
        used = np.zeros(len(uv), bool)
        if len(n):
            cost, p = _pair_costs(A, b, c, n, Aj, bj, cj)
            bad = ~np.isfinite(cost) | (cost > tol ** 2) | ~np.isfinite(p[..., 2]) | (p[..., 2] < hmin) | (p[..., 2] > hmax)
            r, q = linear_sum_assignment(np.where(bad, 1e6, cost))
            for k, m in zip(r, q):
                if bad[k, m]:
                    continue
                A[k] += Aj[m]
                b[k] += bj[m]
                c[k] += cj[m]
                n[k] += 1
                ssum[k] += sc[m]
                used[m] = True
        new = ~used
        if new.any():
            A = np.concatenate([A, Aj[new]])
            b = np.concatenate([b, bj[new]])
            c = np.concatenate([c, cj[new]])
            n = np.concatenate([n, np.ones(int(new.sum()))])
            ssum = np.concatenate([ssum, sc[new]])
            sxy = np.concatenate([sxy, xy1[new][:, :2]])
    if len(n) == 0:
        return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0), np.zeros(0)
    xy = sxy.copy()
    h = np.full(len(n), single_h)
    multi = n >= 2
    if multi.any():
        p = np.linalg.solve(A[multi] + 1e-9 * np.eye(3), b[multi][..., None])[..., 0]
        xy[multi] = p[:, :2]
        h[multi] = p[:, 2]
    keep = np.ones(len(n), bool)
    if bounds is not None:
        keep = (xy[:, 0] > bounds[0]) & (xy[:, 0] < bounds[1]) & (xy[:, 1] > bounds[2]) & (xy[:, 1] < bounds[3])
    return xy[keep], n[keep].astype(int), (ssum / n)[keep], h[keep]


def ground_density(points: np.ndarray, grid: GroundGrid, sigma_m: float = 0.6) -> np.ndarray:
    ny, nx = grid.shape
    d = np.zeros((ny, nx), dtype=np.float32)
    if len(points):
        ix = np.floor((points[:, 0] - grid.x_min) / grid.cell).astype(int)
        iy = np.floor((points[:, 1] - grid.y_min) / grid.cell).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        np.add.at(d, (iy[ok], ix[ok]), 1.0)
    return gaussian_filter(d, sigma_m / grid.cell) / (grid.cell ** 2)


def match_points(pred: np.ndarray, gt: np.ndarray, radius: float):
    if len(pred) == 0 or len(gt) == 0:
        return 0, len(pred), len(gt), np.zeros(0)
    D = np.hypot(pred[:, None, 0] - gt[None, :, 0], pred[:, None, 1] - gt[None, :, 1])
    r, c = linear_sum_assignment(D)
    good = D[r, c] <= radius
    tp = int(good.sum())
    return tp, len(pred) - tp, len(gt) - tp, D[r, c][good]


# --- 5. IMAGE-SPACE EVALUATION ---
def match_boxes_to_points(boxes: np.ndarray, pts: np.ndarray, margin: float = 0.10):
    if len(boxes) == 0 or len(pts) == 0:
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
    rel = [(pts[j, 1] - boxes[i, 1]) / max(boxes[i, 3] - boxes[i, 1], 1.0) for i, j in m]
    return {"tp": len(m), "fp": len(boxes) - len(m), "fn": len(pts) - len(m),
            "rel_y": float(np.median(rel)) if rel else float("nan")}


def prf(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# --- 6. VISUALS ---
def draw_view(img: np.ndarray, boxes: np.ndarray, pts: Optional[np.ndarray] = None, scale: float = 0.5) -> np.ndarray:
    out = img.copy()
    for (x0, y0, x1, y1) in boxes:
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 220, 0), 3)
    if pts is not None:
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), 9, (0, 0, 255), 3)
    return cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def tile_views(imgs: List[np.ndarray], target_h: int = 540) -> np.ndarray:
    tiles = [cv2.resize(i, (int(i.shape[1] * target_h / i.shape[0]), target_h)) for i in imgs]
    if len(tiles) == 1:
        return tiles[0]
    rows = []
    for k in range(0, len(tiles), 2):
        pair = tiles[k:k + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        rows.append(np.hstack(pair))
    W = max(r.shape[1] for r in rows)
    return np.vstack([np.hstack([r, np.zeros((r.shape[0], W - r.shape[1], 3), np.uint8)]) if r.shape[1] < W else r
                      for r in rows])


def _dark(ax):
    ax.set_facecolor("#111111")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")


def plot_result(path: str, overlays: List[np.ndarray], grid: GroundGrid, fused: np.ndarray, nviews: np.ndarray,
                density: Optional[np.ndarray], count: int, gt: Optional[np.ndarray] = None,
                n_boxes: int = 0, dpi: int = 140):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        _dark(ax)
    if overlays:
        axes[0].imshow(cv2.cvtColor(tile_views(overlays), cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"YOLO persons | {len(overlays)} cam(s) | {n_boxes} boxes", color="white", fontsize=10, pad=6)
    axes[0].axis("off")
    ax = axes[1]
    if density is None:
        ax.text(0.5, 0.5, "no calibrated camera\n(ground plane unavailable)", color="white", ha="center", va="center",
                transform=ax.transAxes)
        axes[2].axis("off")
    else:
        if gt is not None and len(gt):
            ax.scatter(gt[:, 0], gt[:, 1], marker="x", c="#ff0055", s=36, label=f"GT ({len(gt)})", zorder=4)
        if len(fused):
            sc = ax.scatter(fused[:, 0], fused[:, 1], c=nviews, cmap="viridis", vmin=1, vmax=3, s=30,
                            edgecolors="white", linewidths=0.5, label=f"YOLO ({len(fused)})", zorder=5)
            cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("views supporting", color="#aaaaaa", fontsize=8)
            cb.ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.set_xlim(grid.x_min, grid.x_max)
        ax.set_ylim(grid.y_max, grid.y_min)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", color="#aaaaaa", fontsize=8)
        ax.set_ylabel("Y (m, down)", color="#aaaaaa", fontsize=8)
        ax.legend(loc="upper right", facecolor="#222222", edgecolor="none", labelcolor="white", fontsize=8)
        ax.set_title("Ground-plane detections", color="white", fontsize=10, pad=6)
        im = axes[2].imshow(density, extent=(grid.x_min, grid.x_max, grid.y_max, grid.y_min), cmap="inferno",
                            aspect="equal")
        cb = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cb.set_label("persons / m\u00b2", color="#aaaaaa", fontsize=8)
        cb.ax.tick_params(colors="#aaaaaa", labelsize=8)
        axes[2].set_title(f"Crowd density | {count} persons", color="white", fontsize=10, pad=6)
        axes[2].set_xlabel("X (m)", color="#aaaaaa", fontsize=8)
        axes[2].set_ylabel("Y (m, down)", color="#aaaaaa", fontsize=8)
    plt.tight_layout(pad=1.5)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)


# --- 7. DATASET + EVAL CLI ---
class CityStreetData:
    def __init__(self, root: str):
        j = os.path.join
        self.labels_dir = j(root, "CityStreet_correctedlabels")
        self.images_dir = j(root, "CityStreet_zip", "CityStreet", "CityStreet", "image_frames")
        self.roi_dir = j(root, "CityStreet_zip", "CityStreet", "CityStreet", "ROI_maps", "ROIs", "camera_view")
        self.gt_h5 = j(self.labels_dir, "Street_groundplane_pmap.h5")
        self.cameras = create_citystreet_cameras()
        self.roi = {v: load_roi_mask(self.roi_dir, i + 1) for i, v in enumerate(self.cameras)}
        self._gt = None
        self._via: Dict[str, Dict[int, np.ndarray]] = {}

    def load_frame_images(self, fid: int) -> Dict[str, np.ndarray]:
        out = {}
        for v, d in VIEW_TO_CAMERA_DIR.items():
            p = os.path.join(self.images_dir, d, f"frame_{fid:04d}.jpg")
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(p)
            out[v] = img
        return out

    def gt(self) -> np.ndarray:
        if self._gt is None:
            self._gt = load_gt_groundplane(self.gt_h5) if os.path.exists(self.gt_h5) else np.zeros((0, 5))
        return self._gt

    def gt_points(self, fid: int) -> np.ndarray:
        return gt_points_for_frame(self.gt(), fid)

    def via_points(self, view: str) -> Dict[int, np.ndarray]:
        if view not in self._via:
            p = os.path.join(self.labels_dir, f"via_region_data_{view}.json")
            self._via[view] = load_via_points(p) if os.path.exists(p) else {}
        return self._via[view]


def pick_frames(ds: CityStreetData, arg: str, n: int) -> List[int]:
    if arg:
        if ":" in arg:
            a = [int(x) for x in arg.split(":")]
            return list(range(a[0], a[1] + 1, a[2] if len(a) > 2 else 2))
        return [int(x) for x in arg.split(",") if x.strip()]
    sets = [set(ds.via_points(v).keys()) for v in ds.cameras]
    common = sorted(set.intersection(*sets) & set(int(f) for f in np.unique(ds.gt()[:, 0])))
    return common[:: max(1, len(common) // max(n, 1))][:n]


def cmd_eval(a):
    cfg = YoloConfig(model=a.model, device=a.device, conf=a.sweep_min, imgsz_full=a.imgsz, tile=a.tile,
                     tile_overlap=a.overlap, tile_imgsz=a.tile_imgsz, use_full=not a.no_full,
                     use_tiles=not a.no_tiles, augment=a.augment)
    ds = CityStreetData(a.data_root)
    frames = pick_frames(ds, a.frames, a.num_frames)
    print(f"frames: {frames} | model {cfg.model} | full={cfg.use_full}@{cfg.imgsz_full} "
          f"tiles={cfg.use_tiles}@{cfg.tile}/{cfg.tile_overlap}")
    det = PersonDetector(cfg)
    os.makedirs(a.output_dir, exist_ok=True)
    cache, t0 = {}, time.time()
    for fid in frames:
        for v, im in ds.load_frame_images(fid).items():
            t1 = time.time()
            cache[(fid, v)] = det.detect(im, ds.roi.get(v))
            print(f"  frame {fid} {v}: {len(cache[(fid, v)][0])} boxes @conf>={cfg.conf} ({time.time() - t1:.1f}s)")
    print(f"detection total {time.time() - t0:.1f}s")

    have = {(f, v): ds.via_points(v).get(f) for f in frames for v in ds.cameras}
    have = {k: p for k, p in have.items() if p is not None and len(p)}
    thr, best = a.conf if a.conf is not None else 0.20, (-1.0, a.sweep_min)
    if have:
        print("\nIMAGE-SPACE EVALUATION vs VIA labels (all views, all frames)")
        print(" conf   boxes  labels    TP    FP    FN   Prec  Rec    F1")
        rows = []
        for t in [round(x, 2) for x in np.arange(a.sweep_min, 0.71, 0.05)]:
            tp = fp = fn = nb = 0
            for k, pts in have.items():
                b, s = cache[k]
                m = eval_view(b[s >= t], pts)
                tp, fp, fn, nb = tp + m["tp"], fp + m["fp"], fn + m["fn"], nb + int((s >= t).sum())
            p, r, f1 = prf(tp, fp, fn)
            rows.append((t, nb, tp + fn, tp, fp, fn, p, r, f1))
            best = max(best, (f1, t))
            print(f" {t:4.2f}  {nb:6d} {tp + fn:7d} {tp:5d} {fp:5d} {fn:5d}  {p:.3f} {r:.3f} {f1:.3f}")
        if a.conf is None:
            thr = best[1]
        print(f"\nbest F1 {best[0]:.3f} at conf {best[1]} | using conf {thr}")
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
        print("\nno VIA labels for these frames; using conf", thr)

    print(f"\nGROUND-PLANE EVALUATION (conf >= {thr}, fuse radius {a.fuse_radius} m, GT match radius {a.match_radius} m)")
    print("  projection  min_views |   TP   FP   FN   Prec  Rec    F1 | count MAE | median loc err")
    for mode in ("head", "feet"):
        acc = {mv: [0, 0, 0, [], []] for mv in (1, 2)}
        for fid in frames:
            gt = ds.gt_points(fid)
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
        for mv, r in acc.items():
            p, rc, f1 = prf(r[0], r[1], r[2])
            print(f"  {mode:10s}  {mv:9d} | {r[0]:4d} {r[1]:4d} {r[2]:4d}  {p:.3f} {rc:.3f} {f1:.3f} | "
                  f"{np.mean(r[4]) if r[4] else float('nan'):9.1f} | {np.median(r[3]) if r[3] else float('nan'):.2f} m")
    for fid in frames[: a.save_frames]:
        for v, im in ds.load_frame_images(fid).items():
            b, s = cache[(fid, v)]
            cv2.imwrite(os.path.join(a.output_dir, f"yolo_{fid}_{v}.jpg"), draw_view(im, b[s >= thr], have.get((fid, v))))
    print(f"outputs in {a.output_dir}/")


def main(argv=None):
    ap = argparse.ArgumentParser(description="YOLO person detection benchmark on CityStreet")
    ap.add_argument("command", choices=["eval"])
    ap.add_argument("--data-root", default="CityStreet")
    ap.add_argument("--frames", default="")
    ap.add_argument("--num-frames", type=int, default=3)
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--tile-imgsz", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--no-full", action="store_true")
    ap.add_argument("--no-tiles", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--conf", type=float, default=None, help="final threshold (default: best F1 from sweep)")
    ap.add_argument("--sweep-min", type=float, default=0.05)
    ap.add_argument("--fuse-radius", type=float, default=1.0)
    ap.add_argument("--match-radius", type=float, default=1.0)
    ap.add_argument("--save-frames", type=int, default=1)
    ap.add_argument("--output-dir", default="yolo_outputs")
    cmd_eval(ap.parse_args(argv))


if __name__ == "__main__":
    main()