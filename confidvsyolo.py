from __future__ import annotations

import argparse
import csv
import os
import sys
import types

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_yolo as ev
import matplotlib.pyplot as plt
from crowd3d.crowd3d_all import CityStreetData, YoloConfig

METHODS = ["YOLO (raw)", "Adaptive confidence", "Ours"]
BGR = {"red": (60, 60, 235), "green": (80, 200, 80), "yellow": (0, 215, 255), "cyan": (230, 200, 40)}
LOW_T, HIGH_T = 0.25, 0.50


def image_quality(gray):
    blur_q = float(np.clip(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500.0, 0.0, 1.0))
    bright_q = float(np.clip(1.0 - abs(float(gray.mean()) - 127.5) / 127.5, 0.0, 1.0))
    contrast_q = float(np.clip(float(gray.std()) / 64.0, 0.0, 1.0))
    return 0.40 * blur_q + 0.30 * bright_q + 0.30 * contrast_q


def adaptive_estimate(scores, quality):
    low = [s for s in scores if s >= LOW_T]
    high = [s for s in scores if s >= HIGH_T]
    mean_low = float(np.mean(low)) if low else 0.0
    mean_high = float(np.mean(high)) if high else 0.0
    w_low = max(0.01, 0.70 * mean_low + 0.30 * quality)
    w_high = max(0.01, 0.70 * mean_high + 0.30 * quality)
    fused = (len(low) * w_low + len(high) * w_high) / (w_low + w_high)
    return int(round(fused))


def roi_keep(boxes, roi):
    if roi is None or len(boxes) == 0:
        return np.ones(len(boxes), bool)
    h, w = roi.shape[:2]
    fx = np.clip(((boxes[:, 0] + boxes[:, 2]) / 2).astype(int), 0, w - 1)
    fy = np.clip(boxes[:, 3].astype(int), 0, h - 1)
    return roi[fy, fx] > 0


def colab_detect(model, img, args):
    kw = dict(conf=args.colab_conf, iou=0.50, imgsz=args.colab_imgsz, augment=not args.no_augment, verbose=False)
    if args.device:
        kw["device"] = args.device
    r = model(img, **kw)[0]
    cls = r.boxes.cls.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    xyxy = r.boxes.xyxy.cpu().numpy()
    m = cls == 0
    return xyxy[m].astype(np.float32), conf[m].astype(np.float32)


def load_colab_model(name):
    from ultralytics import YOLO
    return YOLO(name)


def draw(small, scale, boxes, colors, legend):
    out = small.copy()
    for b, c in zip(boxes, colors):
        x1, y1, x2, y2 = [int(round(v * scale)) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), BGR[c], 2)
    for i, (txt, c) in enumerate(legend):
        cv2.putText(out, txt, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BGR[c], 2, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def metrics(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    err = np.abs(pred - truth)
    ok = truth > 0
    return dict(MAE=float(err.mean()), NAE=float((err[ok] / truth[ok]).mean()),
                RMSE=float(np.sqrt(((pred - truth) ** 2).mean())), bias=float((pred - truth).mean()))


def run(args, ds, frames, views, L, dets_ours, model):
    rows = []
    for fid in frames:
        imgs = ds.load_frame_images(fid)
        for v in views:
            img = imgs[v]
            roi = ds.roi.get(v)
            truth = int(L[(fid, v)]["inroi"].sum())
            b, s = colab_detect(model, img, args)
            keep = roi_keep(b, roi)
            b, s = b[keep], s[keep]
            quality = image_quality(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            ob, os_ = dets_ours[(fid, v)]
            sel = os_ >= args.ours_conf
            ob, os_ = ob[sel], os_[sel]
            scale = args.max_width / img.shape[1]
            small = cv2.resize(img, (args.max_width, int(round(img.shape[0] * scale))))
            rows.append(dict(frame=fid, view=v, truth=truth, quality=quality, scale=scale, small=small,
                             preds=[len(s), adaptive_estimate(s, quality), len(os_)],
                             yolo_boxes=b, yolo_scores=s, ours_boxes=ob))
            print(f"frame {fid} {v}: true {truth} | YOLO {len(s)} | adaptive {rows[-1]['preds'][1]} | ours {len(os_)}", flush=True)
    return rows


def render(rows, summary, out_dir):
    n = len(rows)
    for r in rows:
        fig, axes = plt.subplots(1, 4, figsize=(26, 4.8))
        axes[0].imshow(cv2.cvtColor(r["small"], cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original | frame {r['frame']} {r['view']} | true count {r['truth']}", fontsize=11)
        y_boxes = list(r["yolo_boxes"])
        high = [s >= HIGH_T for s in r["yolo_scores"]]
        low = [s >= LOW_T for s in r["yolo_scores"]]
        panels = [
            draw(r["small"], r["scale"], y_boxes, ["red"] * len(y_boxes), [(f"all boxes >= {ARGS.colab_conf:.2f}", "red")]),
            draw(r["small"], r["scale"], [b for b, l in zip(y_boxes, low) if l],
                 ["green" if h else "yellow" for h, l in zip(high, low) if l],
                 [(f"green >= {HIGH_T:.2f}", "green"), (f"yellow {LOW_T:.2f}-{HIGH_T:.2f}", "yellow")]),
            draw(r["small"], r["scale"], list(r["ours_boxes"]), ["cyan"] * len(r["ours_boxes"]), [(f"boxes >= {ARGS.ours_conf:.2f}", "cyan")]),
        ]
        for k, name in enumerate(METHODS):
            ax = axes[k + 1]
            ax.imshow(panels[k])
            pred, truth = r["preds"][k], r["truth"]
            err = abs(pred - truth)
            nae = err / truth if truth > 0 else float("nan")
            ax.set_title(f"{name} | predicted {pred}", fontsize=11)
            ax.set_xlabel(f"this image: abs error {err} | NAE {nae:.3f}\nwhole set ({n} images): MAE {summary[name]['MAE']:.2f} | NAE {summary[name]['NAE']:.3f}", fontsize=11)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"count_{r['frame']}_{r['view']}.png"), dpi=70)
        plt.close(fig)


def render_summary(rows, summary, out_dir):
    truth = np.array([r["truth"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cols = ["#d9534f", "#e8b31d", "#2a9fd6"]
    for ax, key in zip(axes[:2], ("MAE", "NAE")):
        vals = [summary[m][key] for m in METHODS]
        bars = ax.bar(METHODS, vals, color=cols)
        for b_, v_ in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v_, f"{v_:.2f}" if key == "MAE" else f"{v_:.3f}", ha="center", va="bottom", fontsize=11)
        ax.set_title(f"{key} over {len(rows)} images (lower is better)")
        ax.tick_params(axis="x", labelrotation=12)
    for k, m in enumerate(METHODS):
        axes[2].scatter(truth, [r["preds"][k] for r in rows], label=m, color=cols[k], alpha=0.8)
    lim = [0, max(truth.max(), max(max(r["preds"]) for r in rows)) * 1.05]
    axes[2].plot(lim, lim, "k--", linewidth=1)
    axes[2].set_xlabel("true count")
    axes[2].set_ylabel("predicted count")
    axes[2].set_title("predicted vs true")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "count_summary.png"), dpi=90)
    plt.close(fig)


def main():
    global ARGS
    ap = argparse.ArgumentParser(description="Compare raw YOLO, the adaptive-confidence method and our pipeline against true counts")
    ap.add_argument("--data-root", default=ev.DATA_ROOT)
    ap.add_argument("--frames", default="", help="comma list of frame numbers; default is spread-out test frames")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--views", default="view1,view2,view3")
    ap.add_argument("--colab-model", default="yolov8n.pt")
    ap.add_argument("--colab-imgsz", type=int, default=1280)
    ap.add_argument("--colab-conf", type=float, default=0.10)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--ours-conf", type=float, default=0.10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-width", type=int, default=900)
    ap.add_argument("--out", default=os.path.join(ev.OUT_DIR, "count_compare"))
    args = ap.parse_args()
    ARGS = args

    ds = CityStreetData(args.data_root)
    if args.frames:
        frames = [int(x) for x in args.frames.split(",") if x.strip()]
    else:
        ns = types.SimpleNamespace(tune_frames="", n_tune=0, test_frames="", n_test=args.n_frames)
        _, frames, _ = ev.pick_frames(ds, ns)
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    print(f"{len(frames)} frames x {len(views)} views = {len(frames) * len(views)} images | frames {frames}")
    print("true count = labelled people inside each camera's region of interest; every method counts only inside that region")

    base = dict(ev.YOLO)
    if args.device:
        base["device"] = args.device
    cfg = YoloConfig(**{**base, "conf": ev.SWEEP_MIN})
    dets_ours, _ = ev.detect_all(ds, cfg, frames, not args.no_cache)
    L = ev.build_labels(ds, frames)
    model = load_colab_model(args.colab_model)
    rows = run(args, ds, frames, views, L, dets_ours, model)

    truth = [r["truth"] for r in rows]
    summary = {m: metrics([r["preds"][k] for r in rows], truth) for k, m in enumerate(METHODS)}
    os.makedirs(args.out, exist_ok=True)
    render(rows, summary, args.out)
    render_summary(rows, summary, args.out)
    with open(os.path.join(args.out, "count_compare.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "view", "true", "yolo_raw", "adaptive", "ours", "image_quality"])
        for r in rows:
            w.writerow([r["frame"], r["view"], r["truth"], *r["preds"], f"{r['quality']:.3f}"])

    print("\nRESULTS (counts per camera image; MAE and RMSE in people, NAE = abs error / true count, bias > 0 means over-counting)")
    print(f"  {'method':20s} {'MAE':>7s} {'NAE':>7s} {'RMSE':>7s} {'bias':>7s}")
    for m in METHODS:
        s = summary[m]
        print(f"  {m:20s} {s['MAE']:7.2f} {s['NAE']:7.3f} {s['RMSE']:7.2f} {s['bias']:+7.2f}")
    print(f"\nfigures: {args.out}/count_<frame>_<view>.png (original + 3 annotated panels with errors), count_summary.png, count_compare.csv")


if __name__ == "__main__":
    main()