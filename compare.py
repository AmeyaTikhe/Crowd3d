from __future__ import annotations

import argparse
import csv
import os
import importlib
import sys
import tempfile
import time
import types
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_yolo as ev
import matplotlib.pyplot as plt
from crowd3d.crowd3d_all import VIEW_TO_CAMERA_DIR, CityStreetData, YoloConfig

BASE_METHODS = ["YOLO (raw)", "Adaptive confidence", "Ours"]
CSR_NAME = "CSRNet (SHA)"
BGR = {"red": (60, 60, 235), "green": (80, 200, 80), "yellow": (0, 215, 255), "cyan": (230, 200, 40)}
BAR_COLORS = ["#d9534f", "#e8b31d", "#2a9fd6", "#6f42c1"]
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
    return int(round((len(low) * w_low + len(high) * w_high) / (w_low + w_high)))


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


def csrnet_device(args):
    import torch
    return "cuda" if (args.device and args.device != "cpu" and torch.cuda.is_available()) else "cpu"


def patch_lwcc_weights(folder):
    from lwcc.util.functions import build_url
    mod = importlib.import_module("lwcc.models.CSRNet")

    def weights_check(model_name, model_weights):
        os.makedirs(folder, exist_ok=True)
        file_name = f"{model_name}_{model_weights}.pth"
        out = os.path.join(folder, file_name)
        if not os.path.isfile(out):
            url = build_url(file_name)
            print(f"{file_name} will be downloaded to {out}")
            tmp = out + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, out)
        return out

    mod.weights_check = weights_check


def load_csrnet(args):
    from lwcc import LWCC
    patch_lwcc_weights(args.lwcc_weights_dir)
    model = LWCC.load_model(model_name="CSRNet", model_weights="SHA")
    model.eval()
    model.to(csrnet_device(args))
    return model


def csrnet_density(model, path, args):
    import torch
    from lwcc import LWCC
    tensor, _ = LWCC.load_image(path, model_name="CSRNet", resize_img=not args.csrnet_no_resize)
    with torch.inference_mode():
        out = model(tensor.to(csrnet_device(args)))
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim != 4:
        raise ValueError(f"Unexpected CSRNet output shape: {tuple(out.shape)}")
    density = out[0, 0].detach().cpu().numpy()
    if not np.isfinite(density).all():
        raise ValueError("CSRNet returned invalid density values")
    return density


def frame_path(ds, fid, v, img):
    p = os.path.join(getattr(ds, "images_dir", ""), VIEW_TO_CAMERA_DIR.get(v, ""), f"frame_{fid:04d}.jpg")
    if os.path.isfile(p):
        return p, False
    fd, tmp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(tmp, img)
    return tmp, True


def csrnet_estimate(density, roi, size):
    w, h = size
    density = np.maximum(density.astype(np.float32), 0.0)
    total = float(density.sum())
    d_up = cv2.resize(density, (w, h), interpolation=cv2.INTER_LINEAR)
    s_up = max(float(d_up.sum()), 1e-9)
    d_up = d_up * (total / s_up)
    if roi is None:
        return total, d_up
    roi_s = cv2.resize((roi > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    masked = np.where(roi_s, d_up, 0.0)
    return float(masked.sum()), masked


def draw(small, scale, boxes, colors, legend):
    out = small.copy()
    for b, c in zip(boxes, colors):
        x1, y1, x2, y2 = [int(round(v * scale)) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), BGR[c], 2)
    for i, (txt, c) in enumerate(legend):
        cv2.putText(out, txt, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BGR[c], 2, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def draw_density(small, density):
    out = small.copy()
    pos = density[density > 0]
    top = float(np.percentile(pos, 99)) if pos.size else 1.0
    norm = np.clip(density / max(top, 1e-9), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    m = norm > 0.04
    out[m] = (0.45 * out[m] + 0.55 * heat[m]).astype(np.uint8)
    cv2.putText(out, "density heatmap (inside region)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def metrics(pred, truth):
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    err = np.abs(pred - truth)
    ok = truth > 0
    return dict(MAE=float(err.mean()), NAE=float((err[ok] / truth[ok]).mean()),
                RMSE=float(np.sqrt(((pred - truth) ** 2).mean())), bias=float((pred - truth).mean()))


def run(args, ds, frames, views, L, dets_ours, model, csr):
    rows, csr_times = [], []
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
            size = (args.max_width, int(round(img.shape[0] * scale)))
            small = cv2.resize(img, size)
            preds = [len(s), adaptive_estimate(s, quality), len(os_)]
            density = None
            if csr is not None:
                path, tmp = frame_path(ds, fid, v, img)
                t0 = time.perf_counter()
                dens = csrnet_density(csr, path, args)
                csr_times.append(time.perf_counter() - t0)
                if tmp:
                    os.remove(path)
                count, density = csrnet_estimate(dens, roi, size)
                preds.append(max(0, int(round(count))))
            rows.append(dict(frame=fid, view=v, truth=truth, quality=quality, scale=scale, small=small, preds=preds,
                             yolo_boxes=b, yolo_scores=s, ours_boxes=ob, density=density))
            print(f"frame {fid} {v}: true {truth} | " + " | ".join(f"{n} {p}" for n, p in zip(ARGS_METHODS, preds)), flush=True)
    return rows, (float(np.mean(csr_times)) if csr_times else None)


def render(rows, methods, summary, out_dir):
    n = len(rows)
    for r in rows:
        fig, axes = plt.subplots(1, 1 + len(methods), figsize=(6.5 * (1 + len(methods)), 4.8))
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
        if r["density"] is not None:
            panels.append(draw_density(r["small"], r["density"]))
        for k, name in enumerate(methods):
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


def render_summary(rows, methods, summary, out_dir):
    truth = np.array([r["truth"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    cols = BAR_COLORS[:len(methods)]
    for ax, key in zip(axes[:2], ("MAE", "NAE")):
        vals = [summary[m][key] for m in methods]
        bars = ax.bar(methods, vals, color=cols)
        for b_, v_ in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v_, f"{v_:.2f}" if key == "MAE" else f"{v_:.3f}", ha="center", va="bottom", fontsize=11)
        ax.set_title(f"{key} over {len(rows)} images (lower is better)")
        ax.tick_params(axis="x", labelrotation=15)
    for k, m in enumerate(methods):
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
    global ARGS, ARGS_METHODS
    ap = argparse.ArgumentParser(description="Compare raw YOLO, the adaptive-confidence method, our pipeline and CSRNet against true counts")
    ap.add_argument("--data-root", default=ev.DATA_ROOT)
    ap.add_argument("--frames", default="", help="comma list of frame numbers; default is spread-out test frames")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--views", default="view1,view2,view3")
    ap.add_argument("--colab-model", default="yolov8n.pt")
    ap.add_argument("--colab-imgsz", type=int, default=1280)
    ap.add_argument("--colab-conf", type=float, default=0.10)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--ours-conf", type=float, default=0.10)
    ap.add_argument("--no-csrnet", action="store_true", help="skip the CSRNet (lwcc) method")
    ap.add_argument("--lwcc-weights-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "lwcc_weights"),
                    help="where the CSRNet weights are stored (default: a lwcc_weights folder next to this script)")
    ap.add_argument("--csrnet-no-resize", action="store_true", help="run CSRNet at full resolution (needs a lot of memory)")
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

    csr = None
    if not args.no_csrnet:
        try:
            csr = load_csrnet(args)
        except Exception as e:
            sys.exit(f"\nCould not load CSRNet ({type(e).__name__}: {e}).\n"
                     "If lwcc is missing: pip install lwcc torch torchvision gdown. The weights download from GitHub on the first run;\n"
                     "if that fails, download https://github.com/tersekmatija/lwcc_weights/releases/download/v0.1/CSRNet_SHA.pth\n"
                     f"in a browser and save it as {os.path.join(args.lwcc_weights_dir, 'CSRNet_SHA.pth')}\n"
                     "To run without CSRNet on purpose, add --no-csrnet.")
    methods = BASE_METHODS + ([CSR_NAME] if csr is not None else [])
    ARGS_METHODS = methods
    print(f"methods compared ({len(methods)}): " + ", ".join(methods))

    base = dict(ev.YOLO)
    if args.device:
        base["device"] = args.device
    cfg = YoloConfig(**{**base, "conf": ev.SWEEP_MIN})
    dets_ours, _ = ev.detect_all(ds, cfg, frames, not args.no_cache)
    L = ev.build_labels(ds, frames)
    model = load_colab_model(args.colab_model)
    rows, csr_seconds = run(args, ds, frames, views, L, dets_ours, model, csr)

    truth = [r["truth"] for r in rows]
    summary = {m: metrics([r["preds"][k] for r in rows], truth) for k, m in enumerate(methods)}
    os.makedirs(args.out, exist_ok=True)
    render(rows, methods, summary, args.out)
    render_summary(rows, methods, summary, args.out)
    with open(os.path.join(args.out, "count_compare.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "view", "true"] + methods + ["image_quality"])
        for r in rows:
            w.writerow([r["frame"], r["view"], r["truth"], *r["preds"], f"{r['quality']:.3f}"])

    print(f"\nMETHODS COMPARED ({len(methods)}): " + ", ".join(methods))
    print("RESULTS (counts per camera image; MAE and RMSE in people, NAE = abs error / true count, bias > 0 means over-counting)")
    print(f"  {'method':20s} {'MAE':>7s} {'NAE':>7s} {'RMSE':>7s} {'bias':>7s}")
    for m in methods:
        s = summary[m]
        print(f"  {m:20s} {s['MAE']:7.2f} {s['NAE']:7.3f} {s['RMSE']:7.2f} {s['bias']:+7.2f}")
    if csr_seconds is not None:
        print(f"  CSRNet mean inference time: {csr_seconds:.2f} s per image")
    print(f"\nfigures: {args.out}/count_<frame>_<view>.png (original + annotated panels with errors), count_summary.png, count_compare.csv")


if __name__ == "__main__":
    main()