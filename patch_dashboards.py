"""Patch script — replaces dashboard functions in run_images.py and run_video.py with GridSpec versions."""
import os

# ============================================================
# NEW dashboard for run_images.py
# ============================================================
NEW_IMAGES_DASH = r'''
def _save_dashboard(cameras, grid, detections, recon_res, counting_res, out_path):
    """Save a flexible N-camera dashboard using GridSpec (no colspan)."""
    from matplotlib.gridspec import GridSpec

    n_cams   = len(cameras)
    cam_cols = min(n_cams, 4)
    cam_rows = (n_cams + cam_cols - 1) // cam_cols
    total_rows = cam_rows + 1

    fig = plt.figure(figsize=(6 * cam_cols, 5.5 * total_rows))
    fig.patch.set_facecolor("#1e1e1e")

    gs_top = GridSpec(cam_rows, cam_cols, figure=fig,
                      top=1.0 - 0.04 / total_rows,
                      bottom=1.0 / total_rows + 0.04,
                      hspace=0.25, wspace=0.08)
    gs_bot = GridSpec(1, 2, figure=fig,
                      top=1.0 / total_rows - 0.04,
                      bottom=0.04,
                      hspace=0.20, wspace=0.30)

    pred_pts_m = (
        np.array([[p.x_m, p.y_m, p.z_m] for p in counting_res.detected_persons],
                  dtype=np.float64)
        if counting_res.detected_persons else np.zeros((0, 3))
    )

    # ---- Camera panels ----
    for idx, (vid, cam) in enumerate(cameras.items()):
        row = idx // cam_cols
        col = idx % cam_cols
        ax  = fig.add_subplot(gs_top[row, col])
        ax.set_facecolor("#121212")
        det = detections.get(vid)
        if det is not None and det.overlay_img is not None:
            ax.imshow(cv2.cvtColor(det.overlay_img, cv2.COLOR_BGR2RGB))
            if len(pred_pts_m) > 0:
                uv, val = cam.project(pred_pts_m)
                if val.any():
                    ax.scatter(uv[val, 0], uv[val, 1], s=20, c="red", marker="o",
                               edgecolors="white", linewidths=0.8, zorder=5)
            ax.set_title(f"{vid}  |  Blobs: {len(det.blobs)}",
                         color="white", fontsize=9, fontweight="bold")
        else:
            ax.text(0.5, 0.5, f"{vid}\n[FAILED / NO IMAGE]",
                    ha="center", va="center", color="orange",
                    fontsize=10, transform=ax.transAxes)
            ax.set_title(vid, color="orange", fontsize=9)
        ax.axis("off")

    ext = grid.bev_extent()

    # ---- BEV panel ----
    ax_bev = fig.add_subplot(gs_bot[0, 0])
    ax_bev.set_facecolor("#121212")
    im_bev = ax_bev.imshow(recon_res.ground_occupancy_metric,
                            extent=ext, origin="lower", cmap="magma")
    ax_bev.set_title(
        f"3D Voxel Consensus  |  Cams: {n_cams}  |  Carved: {int(recon_res.occupancy_vol.sum())}",
        color="white", fontsize=10, fontweight="bold")
    ax_bev.set_xlabel("X (m)", color="white", fontsize=8)
    ax_bev.set_ylabel("Y (m)", color="white", fontsize=8)
    ax_bev.tick_params(colors="white", labelsize=7)
    plt.colorbar(im_bev, ax=ax_bev, fraction=0.046, pad=0.04)

    # ---- Density panel ----
    ax_den = fig.add_subplot(gs_bot[0, 1])
    ax_den.set_facecolor("#121212")
    im_den = ax_den.imshow(counting_res.density_map_metric,
                            extent=ext, origin="lower", cmap="inferno")
    if len(pred_pts_m) > 0:
        ax_den.scatter(pred_pts_m[:, 0], pred_pts_m[:, 1],
                       c="cyan", s=20, marker="o", edgecolors="black", linewidths=0.6,
                       label=f"Detected ({counting_res.peak_count})")
    ax_den.set_title(
        f"Ground Density  |  Peak: {counting_res.peak_count}  Mass: {counting_res.mass_count:.1f}",
        color="white", fontsize=10, fontweight="bold")
    ax_den.set_xlabel("X (m)", color="white", fontsize=8)
    ax_den.set_ylabel("Y (m)", color="white", fontsize=8)
    ax_den.tick_params(colors="white", labelsize=7)
    ax_den.legend(loc="upper right", facecolor="#2a2a2a",
                  edgecolor="none", labelcolor="white", fontsize=7)
    plt.colorbar(im_den, ax=ax_den, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Crowd3D Image-Mode  |  {n_cams} Camera(s)  |  "
        f"Peak: {counting_res.peak_count}  Mass: {counting_res.mass_count:.1f}",
        color="white", fontsize=13, fontweight="bold", y=1.01)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved dashboard: {out_path}")

'''

# ============================================================
# NEW dashboard for run_video.py
# ============================================================
NEW_VIDEO_DASH = r'''
def _save_frame_dashboard(frame_idx, cam_states, grid, detections,
                           recon_res, counting_res, current_frames, out_path):
    """Save per-frame N-camera dashboard using GridSpec (no colspan)."""
    from matplotlib.gridspec import GridSpec

    active_cams = {vid: st.camera for vid, st in cam_states.items() if st.active}
    n_cams   = len(active_cams)
    cam_cols = min(max(n_cams, 1), 4)
    cam_rows = max(1, (n_cams + cam_cols - 1) // cam_cols)
    total_rows = cam_rows + 1

    fig = plt.figure(figsize=(6 * cam_cols, 5.5 * total_rows))
    fig.patch.set_facecolor("#1e1e1e")

    gs_top = GridSpec(cam_rows, cam_cols, figure=fig,
                      top=1.0 - 0.04 / total_rows,
                      bottom=1.0 / total_rows + 0.04,
                      hspace=0.25, wspace=0.08)
    gs_bot = GridSpec(1, 2, figure=fig,
                      top=1.0 / total_rows - 0.04,
                      bottom=0.04,
                      hspace=0.20, wspace=0.30)

    pred_pts_m = (
        np.array([[p.x_m, p.y_m, p.z_m] for p in counting_res.detected_persons],
                  dtype=np.float64)
        if counting_res.detected_persons else np.zeros((0, 3))
    )

    # ---- Camera panels ----
    for idx, (vid, cam) in enumerate(active_cams.items()):
        row = idx // cam_cols
        col = idx % cam_cols
        ax  = fig.add_subplot(gs_top[row, col])
        ax.set_facecolor("#121212")
        det = detections.get(vid)
        if det is not None and det.overlay_img is not None:
            ax.imshow(cv2.cvtColor(det.overlay_img, cv2.COLOR_BGR2RGB))
            if len(pred_pts_m) > 0:
                uv, val = cam.project(pred_pts_m)
                if val.any():
                    ax.scatter(uv[val, 0], uv[val, 1], s=16, c="red", marker="o",
                               edgecolors="white", linewidths=0.6, zorder=5)
            ax.set_title(f"{vid}  |  Blobs: {len(det.blobs)}",
                         color="white", fontsize=8, fontweight="bold")
        else:
            img_f = current_frames.get(vid)
            if img_f is not None:
                ax.imshow(cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB))
            ax.set_title(f"{vid} [no BG yet]", color="yellow", fontsize=8)
        ax.axis("off")

    ext = grid.bev_extent()

    # ---- BEV panel ----
    ax_bev = fig.add_subplot(gs_bot[0, 0])
    ax_bev.set_facecolor("#121212")
    im_bev = ax_bev.imshow(recon_res.ground_occupancy_metric,
                            extent=ext, origin="lower", cmap="magma")
    ax_bev.set_title(
        f"Voxel BEV  |  Frame {frame_idx}  |  Active: {n_cams} cam(s)",
        color="white", fontsize=9, fontweight="bold")
    ax_bev.set_xlabel("X (m)", color="white", fontsize=7)
    ax_bev.set_ylabel("Y (m)", color="white", fontsize=7)
    ax_bev.tick_params(colors="white", labelsize=6)
    plt.colorbar(im_bev, ax=ax_bev, fraction=0.046, pad=0.04)

    # ---- Density panel ----
    ax_den = fig.add_subplot(gs_bot[0, 1])
    ax_den.set_facecolor("#121212")
    im_den = ax_den.imshow(counting_res.density_map_metric,
                            extent=ext, origin="lower", cmap="inferno")
    if len(pred_pts_m) > 0:
        ax_den.scatter(pred_pts_m[:, 0], pred_pts_m[:, 1],
                       c="cyan", s=16, marker="o", edgecolors="black", linewidths=0.5,
                       label=f"Peaks ({counting_res.peak_count})")
    ax_den.set_title(
        f"Density  |  Peak: {counting_res.peak_count}  Mass: {counting_res.mass_count:.1f}",
        color="white", fontsize=9, fontweight="bold")
    ax_den.set_xlabel("X (m)", color="white", fontsize=7)
    ax_den.set_ylabel("Y (m)", color="white", fontsize=7)
    ax_den.tick_params(colors="white", labelsize=6)
    ax_den.legend(loc="upper right", facecolor="#2a2a2a",
                  edgecolor="none", labelcolor="white", fontsize=6)
    plt.colorbar(im_den, ax=ax_den, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Crowd3D Video Mode  |  Frame {frame_idx}  |  "
        f"{n_cams} Active Camera(s)  |  Count: {counting_res.peak_count}",
        color="white", fontsize=11, fontweight="bold", y=1.01)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor(),
                edgecolor="none", bbox_inches="tight")
    plt.close(fig)

'''


def patch_file(filename, old_fn_start, old_fn_end, new_body):
    txt = open(filename, encoding="utf-8").read()
    s = txt.index(old_fn_start)
    e = txt.index(old_fn_end, s)
    patched = txt[:s] + new_body + txt[e:]
    open(filename, "w", encoding="utf-8").write(patched)
    print(f"Patched {filename}")


patch_file(
    "run_images.py",
    "\ndef _save_dashboard(",
    "\ndef main(",
    NEW_IMAGES_DASH,
)

patch_file(
    "run_video.py",
    "\ndef _save_frame_dashboard(",
    "\ndef main(",
    NEW_VIDEO_DASH,
)

print("All patches applied.")
