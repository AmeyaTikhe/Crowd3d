from __future__ import annotations

import argparse
import os
import re
import sys
import time
import types
from dataclasses import replace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_yolo as ev
from crowd3d.crowd3d_all import (VIEW_TO_CAMERA_DIR, CityStreetData, GroundGrid, PersonDetector, YoloConfig,
                                 ground_density)

TILE_W, TILE_H, PANEL_W, GROUND_ZOOM = 560, 315, 360, 4
REL_BGR = np.array([(160, 237, 255), (76, 178, 254), (60, 141, 253), (28, 26, 227), (38, 0, 128)], np.uint8)
REL_EDGES = [0.5, 0.75, 0.9, 0.97]
REL_NAMES = ["lowest 50% of occupied area", "next 25%", "next 15%", "next 7%", "hottest 3%"]
POINT_MODES = {"head": ("top", ev.HEAD_H), "mid": ("mid", 0.85), "feet": ("bottom", 0.0)}


def colorize(d, mode, valid, abs_max, ref):
    if mode == "absolute":
        u8 = (np.clip(d / abs_max, 0, 1) * 255).astype(np.uint8)
        return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO), valid & (d >= 0.02)
    occ = valid & (d >= ev.OCCUPIED)
    if ref.size == 0:
        return np.zeros(d.shape + (3,), np.uint8), np.zeros_like(occ)
    band = np.digitize(np.searchsorted(ref, d, side="right") / ref.size, REL_EDGES)
    return REL_BGR[band], occ


def blend(base, bgr, a, alpha):
    out = base.copy()
    out[a] = ((1 - alpha) * base[a] + alpha * bgr[a]).astype(np.uint8)
    return out


def text(img, s, org, scale=0.55, color=(235, 235, 235), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def list_frame_ids(ds, spec, step):
    avail = None
    for d in VIEW_TO_CAMERA_DIR.values():
        p = os.path.join(ds.images_dir, d)
        if not os.path.isdir(p):
            sys.exit(f"Frame folder not found: {p}")
        ids = {int(m.group(1)) for fn in os.listdir(p) for m in [re.match(r"frame_(\d+)\.jpg$", fn)] if m}
        avail = ids if avail is None else avail & ids
    ids = sorted(avail)
    if spec:
        if ":" in spec:
            a, b = [int(x) for x in spec.split(":")[:2]]
            ids = [f for f in ids if a <= f <= b]
        else:
            want = {int(x) for x in spec.split(",") if x.strip()}
            ids = [f for f in ids if f in want]
    return ids[:: max(1, step)]


def dataset_source(ds, ids):
    for fid in ids:
        yield fid, ds.load_frame_images(fid)


def video_source(paths, views, skip):
    caps = [cv2.VideoCapture(p) for p in paths]
    i = 0
    while True:
        imgs = {}
        for v, cap in zip(views, caps):
            ok, im = cap.read()
            if not ok:
                return
            if im.shape[1] != 2704 or im.shape[0] != 1520:
                im = cv2.resize(im, (2704, 1520))
            imgs[v] = im
        if i % max(1, skip) == 0:
            yield i, imgs
        i += 1


class Live:
    def __init__(self, ds, args, get_dets):
        self.ds, self.args, self.get_dets = ds, args, get_dets
        self.grid = GroundGrid()
        g = self.grid
        self.ny, self.nx = g.shape
        self.cell2 = g.cell ** 2
        self.views = list(ds.cameras)
        self.reg = ev.gt_region(ds) if len(ds.gt()) else (g.x_min, g.x_max, g.y_min, g.y_max)
        xs = g.x_min + (np.arange(self.nx) + 0.5) * g.cell
        ys = g.y_min + (np.arange(self.ny) + 0.5) * g.cell
        r = self.reg
        self.mask = (ys[:, None] >= r[2]) & (ys[:, None] <= r[3]) & (xs[None, :] >= r[0]) & (xs[None, :] <= r[1])
        self.which, self.z = POINT_MODES[args.point]
        self.index = {v: self._pixel_index(ds.cameras[v]) for v in self.views}
        self.scale = {v: 1.0 for v in self.views}
        self.maps = {}
        self.mode = args.mode if args.mode != "auto" else "absolute"
        self.history = []

    def _pixel_index(self, cam, stride=8):
        uu, vv = np.meshgrid(np.arange(0, cam.img_w, stride) + stride / 2, np.arange(0, cam.img_h, stride) + stride / 2)
        G, ok = cam.backproject_ground(np.column_stack([uu.ravel(), vv.ravel()]), 0.0)
        with np.errstate(all="ignore"):
            ix = np.floor((G[:, 0] - self.grid.x_min) / self.grid.cell)
            iy = np.floor((G[:, 1] - self.grid.y_min) / self.grid.cell)
        inb = ok & np.isfinite(ix) & np.isfinite(iy) & (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        ix, iy = np.where(inb, ix, 0).astype(int), np.where(inb, iy, 0).astype(int)
        inb &= self.mask[iy, ix]
        return iy.reshape(uu.shape), ix.reshape(uu.shape), inb.reshape(uu.shape)

    def points(self, v, boxes, scores):
        b = boxes[scores >= self.args.conf]
        if not len(b):
            return np.zeros((0, 2))
        y = b[:, 1] if self.which == "top" else (b[:, 3] if self.which == "bottom" else (b[:, 1] + b[:, 3]) / 2)
        p, ok = self.ds.cameras[v].backproject_ground(np.column_stack([(b[:, 0] + b[:, 2]) / 2, y]), self.z)
        xy = p[:, :2][ok]
        return xy[ev.in_region(xy, self.reg)]

    def calibrate(self, tune, dets):
        for v in self.views:
            g = sum(len(self.ds.gt_points(f)) for f in tune)
            p = sum(len(self.points(v, *dets[(f, v)])) for f in tune)
            self.scale[v] = g / p if g and p else 1.0

    def step(self, fid, imgs, elapsed):
        a = self.args
        for v in self.views:
            b, s = self.get_dets(fid, v, imgs[v])
            new = self.scale[v] * ground_density(self.points(v, b, s), self.grid, a.sigma)
            old = self.maps.get(v)
            self.maps[v] = new if old is None or a.smooth <= 0 else (1 - a.smooth) * new + a.smooth * old
        fused = sum(self.maps.values()) / len(self.views)
        occ = self.mask & (fused >= ev.OCCUPIED)
        vals = fused[occ]
        p90 = float(np.percentile(vals, 90)) if vals.size else 0.0
        count = float((fused * self.mask).sum() * self.cell2)
        peak = float((fused * self.mask).max())
        if a.mode == "auto":
            if self.mode == "absolute" and p90 >= 0.72:
                self.mode = "relative"
            elif self.mode == "relative" and p90 < 0.60:
                self.mode = "absolute"
        else:
            self.mode = a.mode
        self.history.append(count)
        gt_n = len(self.ds.gt_points(fid)) if len(self.ds.gt()) else 0
        return self.render(fid, imgs, fused, dict(count=count, p90=p90, peak=peak, gt=gt_n, sec=elapsed))

    def ref_of(self, d):
        return np.sort(d[self.mask & (d >= ev.OCCUPIED)])

    def render(self, fid, imgs, fused, st):
        a, mode = self.args, self.mode
        tiles, originals = [], []
        for v in self.views:
            raw = cv2.resize(imgs[v], (TILE_W, TILE_H))
            o = raw.copy()
            text(o, f"{v} original", (8, 22), 0.6, (255, 255, 255), 2)
            originals.append(o)
            img = raw
            iy, ix, inb = self.index[v]
            d = np.where(inb, self.maps[v][iy, ix], 0.0)
            bgr, m = colorize(d, mode, inb, a.abs_max, self.ref_of(self.maps[v]))
            bgr = cv2.resize(bgr, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)
            m = cv2.resize(m.astype(np.uint8), (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST) > 0
            img = blend(img, bgr, m, 0.6)
            text(img, f"{v} with heatmap", (8, 22), 0.6, (255, 255, 255), 2)
            tiles.append(img)
        height = TILE_H * 3
        left = np.vstack(tiles)
        if not getattr(a, "no_originals", False):
            left = np.hstack([np.vstack(originals), left])

        base = np.full((self.ny, self.nx, 3), 27, np.uint8)
        bgr, m = colorize(fused, mode, self.mask, a.abs_max, self.ref_of(fused))
        ground = cv2.resize(blend(base, bgr, m, 0.95), (self.nx * GROUND_ZOOM, self.ny * GROUND_ZOOM), interpolation=cv2.INTER_LINEAR)
        g = self.grid
        for x in range(int(np.ceil(g.x_min / 10)) * 10, int(g.x_max) + 1, 10):
            px = int((x - g.x_min) / g.cell * GROUND_ZOOM)
            cv2.line(ground, (px, 0), (px, ground.shape[0]), (70, 70, 70), 1)
            text(ground, f"{x} m", (px + 3, ground.shape[0] - 6), 0.4, (150, 150, 150))
        for y in range(int(np.ceil(g.y_min / 10)) * 10, int(g.y_max) + 1, 10):
            py = int((y - g.y_min) / g.cell * GROUND_ZOOM)
            cv2.line(ground, (0, py), (ground.shape[1], py), (70, 70, 70), 1)
            text(ground, f"{y} m", (4, py - 4), 0.4, (150, 150, 150))
        if a.show_gt and len(self.ds.gt()):
            for X, Y in self.ds.gt_points(fid)[:, :2]:
                cv2.circle(ground, (int((X - g.x_min) / g.cell * GROUND_ZOOM), int((Y - g.y_min) / g.cell * GROUND_ZOOM)), 3, (255, 255, 255), -1)
        text(ground, f"fused ground-plane heatmap ({mode} scale)", (8, 22), 0.6, (255, 255, 255), 2)
        pad = max(0, height - ground.shape[0])
        mid = np.vstack([ground, np.full((pad, ground.shape[1], 3), 27, np.uint8)]) if pad else ground[:height]

        panel = np.full((height, PANEL_W, 3), 35, np.uint8)
        y = 30
        for s, sc, col in ((f"frame {fid}", 0.7, (255, 255, 255)), (f"processing {st['sec']:.1f} s/frame", 0.5, (180, 180, 180)),
                           (f"scale: {mode}" + (" (auto)" if a.mode == "auto" else ""), 0.6, (80, 220, 255)),
                           (f"people (fused): {st['count']:.0f}", 0.7, (255, 255, 255)),
                           (f"ground truth: {st['gt']}" if st["gt"] else "ground truth: n/a", 0.55, (180, 180, 180)),
                           (f"busiest 10%: {st['p90']:.2f} /m2  level {ev.los_letter(st['p90'])}", 0.55, (255, 255, 255)),
                           (f"peak: {st['peak']:.2f} /m2", 0.55, (255, 255, 255))):
            text(panel, s, (12, y), sc, col, 1)
            y += 32
        hist = np.array(self.history[-150:])
        text(panel, "people over time", (12, y + 8), 0.5, (180, 180, 180))
        top, bot = y + 20, y + 120
        if len(hist) > 1:
            lo, hi = float(hist.min()), max(float(hist.max()), float(hist.min()) + 1.0)
            pts = np.column_stack([12 + np.arange(len(hist)) * (PANEL_W - 24) / max(len(hist) - 1, 1), bot - (hist - lo) / (hi - lo) * (bot - top)]).astype(np.int32)
            cv2.polylines(panel, [pts], False, (80, 220, 255), 2, cv2.LINE_AA)
            text(panel, f"{hi:.0f}", (PANEL_W - 50, top + 10), 0.4, (150, 150, 150))
            text(panel, f"{lo:.0f}", (PANEL_W - 50, bot), 0.4, (150, 150, 150))
        y = bot + 40
        if mode == "absolute":
            text(panel, f"people per m2 (0 to {a.abs_max:.1f})", (12, y), 0.5, (200, 200, 200))
            bar_top, bar_h = y + 14, 260
            grad = np.linspace(255, 0, bar_h).astype(np.uint8)[:, None]
            bar = cv2.applyColorMap(np.repeat(grad, 36, axis=1), cv2.COLORMAP_TURBO)
            panel[bar_top:bar_top + bar_h, 12:48] = bar
            for i, e in enumerate(ev.LOS_EDGES):
                if e <= a.abs_max:
                    py = int(bar_top + (1 - e / a.abs_max) * bar_h)
                    cv2.line(panel, (12, py), (56, py), (255, 255, 255), 1)
                    text(panel, f"{e:.2f}  {ev.LOS_LETTERS[i]}|{ev.LOS_LETTERS[i + 1]}", (62, py + 4), 0.45, (220, 220, 220))
            text(panel, "0", (62, bar_top + bar_h), 0.45, (220, 220, 220))
        else:
            text(panel, "busier than elsewhere in this frame", (12, y), 0.5, (200, 200, 200))
            ref = self.ref_of(fused)
            for k in range(5):
                py = y + 20 + k * 30
                cv2.rectangle(panel, (12, py), (40, py + 20), tuple(int(c) for c in REL_BGR[k]), -1)
                lo = ref[int(REL_EDGES[k - 1] * ref.size)] if k > 0 and ref.size else 0.0
                text(panel, REL_NAMES[k] + (f"  >= {lo:.2f}" if k > 0 else ""), (48, py + 15), 0.42, (220, 220, 220))
        return np.hstack([left, mid, panel])


def build_detector(args, ds):
    base = dict(ev.YOLO)
    if args.fast:
        base.update(model="yolo11s.pt", use_tiles=False, imgsz_full=960)
    if args.model:
        base["model"] = args.model
    if args.device:
        base["device"] = args.device
    cfg = YoloConfig(**{**base, "conf": ev.SWEEP_MIN})
    tag = ev.cfg_tag(cfg)
    os.makedirs(ev.CACHE_DIR, exist_ok=True)
    holder = {}

    def get_dets(fid, v, img):
        p = os.path.join(ev.CACHE_DIR, f"{tag}_{fid}_{v}.npz")
        if os.path.exists(p):
            d = np.load(p)
            return d["boxes"], d["scores"]
        if "det" not in holder:
            holder["det"] = PersonDetector(cfg)
        b, s = holder["det"].detect(img, ds.roi.get(v))
        np.savez(p, boxes=b, scores=s)
        return b, s

    return cfg, get_dets


def run(args, ds, source, get_dets, calib=None):
    live = Live(ds, args, get_dets)
    if calib:
        live.calibrate(*calib)
        print("count correction per camera: " + ", ".join(f"{v} x{s:.2f}" for v, s in live.scale.items()))
    writer, window_ok, paused, n = None, not args.no_window, False, 0
    period = 1.0 / args.fps if args.fps > 0 else 0.0
    it = iter(source)
    last = None
    while True:
        t0 = time.time()
        if not paused or last is None:
            try:
                fid, imgs = next(it)
            except StopIteration:
                break
            canvas = live.step(fid, imgs, time.time() - t0)
            n += 1
            last = (fid, canvas)
        fid, canvas = last
        if args.display_scale != 1.0:
            canvas = cv2.resize(canvas, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
        if args.save and writer is None:
            writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), max(args.fps, 1.0), (canvas.shape[1], canvas.shape[0]))
        if writer is not None and not paused:
            writer.write(canvas)
        if window_ok:
            try:
                cv2.imshow("crowd heatmap", canvas)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                print("No display available (this OpenCV build has no window support); continuing without a window.")
                window_ok, key = False, 255
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("m") and args.mode == "auto":
                live.mode = "relative" if live.mode == "absolute" else "absolute"
            if key == ord("s"):
                out = os.path.join(ev.OUT_DIR, f"live_snapshot_{fid}.png")
                os.makedirs(ev.OUT_DIR, exist_ok=True)
                cv2.imwrite(out, canvas)
                print(f"saved {out}")
        if args.max_frames and n >= args.max_frames:
            break
        rest = period - (time.time() - t0)
        if rest > 0:
            time.sleep(rest)
    if writer is not None:
        writer.release()
        print(f"video saved to {args.save}")
    if window_ok:
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Live crowd density heatmaps for the three CityStreet cameras")
    ap.add_argument("--data-root", default=ev.DATA_ROOT)
    ap.add_argument("--videos", nargs=3, metavar=("VIEW1", "VIEW2", "VIEW3"), default=None, help="use three synchronised video files instead of the frame folders")
    ap.add_argument("--video-skip", type=int, default=1)
    ap.add_argument("--frames", default=f"{ev.TEST_RANGE[0]}:{ev.TEST_RANGE[1]}", help="START:END or a comma list of frame numbers")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--fps", type=float, default=2.0, help="cap on playback speed; slower if detection is slower")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--point", choices=list(POINT_MODES), default="mid")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--smooth", type=float, default=0.5, help="0 = no temporal smoothing, 0.9 = very slow fade")
    ap.add_argument("--mode", choices=["auto", "absolute", "relative"], default="auto")
    ap.add_argument("--abs-max", type=float, default=1.0)
    ap.add_argument("--dist-model", choices=["tsai", "quartic", "official"], default="tsai")
    ap.add_argument("--fast", action="store_true", help="yolo11s at 960 px with no tiles: much quicker, a bit less accurate")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--show-gt", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--save", default=None, help="write the live view to this .mp4 file")
    ap.add_argument("--display-scale", type=float, default=0.6)
    ap.add_argument("--no-originals", action="store_true", help="hide the column of original camera frames")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    ds = CityStreetData(args.data_root)
    ds.cameras = {v: replace(c, dist_model=args.dist_model) for v, c in ds.cameras.items()}
    cfg, get_dets = build_detector(args, ds)
    calib = None
    if not args.no_calibrate and len(ds.gt()):
        ns = types.SimpleNamespace(tune_frames="", n_tune=4, test_frames="", n_test=8)
        tune, _, _ = ev.pick_frames(ds, ns)
        if tune:
            dets, _ = ev.detect_all(ds, cfg, tune, True)
            calib = (tune, dets)
    if args.videos:
        source = video_source(args.videos, list(ds.cameras), args.video_skip)
    else:
        ids = list_frame_ids(ds, args.frames, args.step)
        if not ids:
            sys.exit("No frames found for that range.")
        print(f"{len(ids)} frames from {ids[0]} to {ids[-1]} | keys: q quit, space pause, m switch scale, s snapshot")
        source = dataset_source(ds, ids)
    run(args, ds, source, get_dets, calib)


if __name__ == "__main__":
    main()