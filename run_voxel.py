"""
run_voxel.py -- 3D voxel crowd-density pipeline with explicit occlusion
reasoning, on one CVCS scene + one synchronised frame.

What it produces
    1. a 3D density volume (sums to the person count, AAAI'20 convention)
    2. per-camera, per-voxel visibility by ray marching through people
       (and optionally through static occluders -- the train-compartment hook)
    3. a coverage report: what fraction of the volume no camera can see, and
       how many people are hidden from every camera
    4. the CVCS distance-map camera-selection weights, per voxel
    5. an annotation-free ray-voting volume -- the live-rig path
    6. plots: BEV, height profile, height slices, coverage map

Usage
    python run_voxel.py --image-root D:/CVCS/images ^
                        --label-root D:/CVCS/labels/100frames_labels_reproduce_640_480_CVCS ^
                        --depth-root D:/100frames_depthMaps_reproduce_640_480 ^
                        --scene scene_02 --frame 1 --views 6 ^
                        --cell-xy 0.5 --cell-z 0.4 --crop 30

    --crop limits the grid to an NxN metre box around the crowd centroid. Start
    small: ray marching is O(n_voxels * n_cams * n_samples).
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from cvcs_checks import check_reprojection, check_world_axis, resolve_pixel_order
from cvcs_io import CVCSPaths, load_view
from cvcs_voxel import (VoxelGrid, camera_weights_from_distance, coverage_report,
                        density_volume_3d, occlusion_weighted_count,
                        ray_vote_volume, visibility_masks)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-root", required=True)
    p.add_argument("--label-root", required=True)
    p.add_argument("--depth-root", required=True)
    p.add_argument("--part", default="train")
    p.add_argument("--scene", default="scene_02")
    p.add_argument("--frame", default="1")
    p.add_argument("--views", type=int, default=6)
    p.add_argument("--cell-xy", type=float, default=0.5, help="CVCS ground resolution")
    p.add_argument("--cell-z", type=float, default=0.4, help="AAAI'20 voxel height")
    p.add_argument("--z-max", type=float, default=2.8, help="AAAI'20 height range")
    p.add_argument("--crop", type=float, default=30.0,
                   help="side of the world box around the crowd centroid, in metres")
    p.add_argument("--tau", type=float, default=1.0, help="opacity scale for ray marching")
    p.add_argument("--samples", type=int, default=48, help="ray-march samples")
    p.add_argument("--out", default="cvcs_voxel_out")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = CVCSPaths(image_root=args.image_root, label_root=args.label_root,
                      depth_root=args.depth_root, part=args.part,
                      scene=args.scene, frame=args.frame)
    all_views = paths.available_views()
    if not all_views:
        raise SystemExit("No complete views found -- check the three root paths.")
    step = max(1, len(all_views) // args.views)
    views = all_views[::step][:args.views]
    print(f"scene={args.scene} frame={args.frame}, using views {views}")

    cams, depths = [], []
    for v in views:
        cam, _, depth = load_view(paths, v, with_image=False,
                                  img_w=1920, img_h=1080)   # set to YOUR jpg size
        resolve_pixel_order(cam, verbose=False)
        check_reprojection(cam)
        cams.append(cam); depths.append(depth)

    # ------------------------------------------------------------- world
    merged = {}
    for cam in cams:
        merged.update(cam.pts_3d)
    print(f"\npeople in frame: {len(merged)}")
    axis = check_world_axis(merged, [c.center for c in cams], verbose=False)
    up = axis["up_axis"]
    print(f"vertical world axis detected: {up}")
    XYZ = np.array(list(merged.values()))
    if up != 2:                      # reorder so Z is up, as the grid assumes
        order = [a for a in range(3) if a != up] + [up]
        XYZ = XYZ[:, order]
        print(f"  reordered world axes to {order}; APPLY THE SAME REORDER to R/t "
              f"before trusting projections (see note at the bottom of this file)")

    # -------------------------------------------------------------- grid
    ctr = np.median(XYZ[:, :2], axis=0)
    h = args.crop / 2.0
    box = XYZ[(np.abs(XYZ[:, 0] - ctr[0]) < h) & (np.abs(XYZ[:, 1] - ctr[1]) < h)]
    grid = VoxelGrid(x0=ctr[0] - h, y0=ctr[1] - h, z0=0.0,
                     cell_xy=args.cell_xy, cell_z=args.cell_z,
                     nx=int(np.ceil(args.crop / args.cell_xy)),
                     ny=int(np.ceil(args.crop / args.cell_xy)),
                     nz=int(np.ceil(args.z_max / args.cell_z)))
    print(f"\nvoxel grid {grid.shape} (nz,ny,nx) = {grid.n_voxels} voxels, "
          f"{grid.cell_xy} m xy / {grid.cell_z} m z, "
          f"{args.crop}x{args.crop}x{args.z_max} m box, {len(box)} people inside")

    # ---------------------------------------------------- 3D density
    density = density_volume_3d(grid, box, sigma_xy=0.3, sigma_z=args.cell_z,
                                body_from_head=True)
    print(f"density volume sum = {density.sum():.2f} (should equal {len(box)})")

    # ------------------------------------------------------ visibility
    print("\nray marching for visibility ...")
    static_occluders = None     # <- put your train-compartment geometry here:
                                #    a (nz,ny,nx) volume, high where seats,
                                #    partitions, racks and walls are. On CVCS
                                #    there is no such structure, so None.
    trans, frust = visibility_masks(grid, cams, occupancy=density,
                                    static_occluders=static_occluders,
                                    n_samples=args.samples, tau=args.tau)

    cov = coverage_report(grid, trans, frust)
    print(f"  voxels outside every camera FOV      : {cov['frac_voxels_unseen']:.1%}")
    print(f"  voxels in FOV but occluded everywhere: {cov['frac_voxels_occluded_everywhere']:.1%}")
    print(f"  voxels seen by >= 2 cameras          : {cov['frac_voxels_multiview']:.1%}")
    print(f"  mean cameras with clear line of sight: {cov['mean_cams_visible']:.2f}")

    occ_count = occlusion_weighted_count(grid, density, trans)
    print(f"\n  true count      : {occ_count['total']:.1f}")
    print(f"  observable      : {occ_count['observable']:.1f}")
    print(f"  hidden from all : {occ_count['hidden']:.1f} "
          f"({occ_count['hidden_fraction']:.1%})")
    print("  ^ this is the number a visibility-blind pipeline silently loses. "
          "Re-run with fewer --views to watch it grow.")

    # --------------------------------------- CVCS camera-selection weights
    W = camera_weights_from_distance(grid, cams, depths)
    print(f"\ncamera-selection weights: shape {W.shape}, "
          f"per-voxel sum in [{W.sum(0).min():.2f}, {W.sum(0).max():.2f}] "
          "(1.0 where any camera sees the voxel, 0 where none do)")

    # --------------------------------------------- annotation-free path
    head_px = {cam.view_id: np.array(list(cam.pts_2d.values())) for cam in cams}
    vote = ray_vote_volume(grid, cams, head_px, sigma_px=12.0)
    print(f"ray-voting volume: max score {vote.max():.2f} "
          f"(peaks should sit where people are; this path uses NO world coords)")

    # ------------------------------------------------------------- plots
    bev = density.sum(axis=0)
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    ext = grid.bev_extent

    im = ax[0, 0].imshow(bev, origin="lower", extent=ext, cmap="inferno")
    ax[0, 0].set_title(f"BEV of 3D density (sum={bev.sum():.0f})")
    plt.colorbar(im, ax=ax[0, 0], fraction=0.046)

    prof = density.sum(axis=(1, 2))
    zc = grid.z0 + (np.arange(grid.nz) + 0.5) * grid.cell_z
    ax[0, 1].barh(zc, prof, height=grid.cell_z * 0.8)
    ax[0, 1].set_title("mass per height level")
    ax[0, 1].set_xlabel("people-equivalent"); ax[0, 1].set_ylabel("height (m)")

    nvis = cov["n_cams_visible"].max(axis=0)
    im = ax[0, 2].imshow(nvis, origin="lower", extent=ext, cmap="viridis")
    ax[0, 2].set_title("cameras with clear line of sight (best height level)")
    plt.colorbar(im, ax=ax[0, 2], fraction=0.046)

    for j, iz in enumerate([0, grid.nz // 2, grid.nz - 1]):
        im = ax[1, j].imshow(density[iz], origin="lower", extent=ext, cmap="inferno")
        ax[1, j].set_title(f"density slice at z={zc[iz]:.2f} m")
        plt.colorbar(im, ax=ax[1, j], fraction=0.046)

    plt.tight_layout()
    out = os.path.join(args.out, f"{args.scene}_frame{args.frame}_voxel.png")
    plt.savefig(out, dpi=130)
    print(f"\nsaved {out}")

    np.savez_compressed(os.path.join(args.out, f"{args.scene}_frame{args.frame}_volume.npz"),
                        density=density, transmittance=trans, frustum=frust,
                        vote=vote, weights=W,
                        grid=np.array([grid.x0, grid.y0, grid.z0, grid.cell_xy,
                                       grid.cell_z, grid.nx, grid.ny, grid.nz]))
    plt.show()


# NOTE on the axis reorder above: if check_world_axis reports that the vertical
# axis is not 2, the honest fix is to rotate the world frame once, at load time,
# and apply the same rotation to R and t (R' = R @ P^T, t unchanged, where P is
# the permutation), rather than permuting only the annotation points. Do not
# ship the half-fix -- it will reproject correctly and place people wrongly.

if __name__ == "__main__":
    main()