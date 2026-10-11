"""
cvcs_voxel.py -- 3D voxel representation with explicit occlusion reasoning.

WHY A VOXEL GRID
----------------
A 2D ground-plane grid cannot represent occlusion: on a single height plane
every person is a zero-thickness point, so nothing is ever behind anything.
A voxel grid gives you somewhere to store, per voxel and per camera:

    in_frustum[k, v]   is voxel v inside camera k's image at all?
    visible[k, v]      is the line of sight from camera k to voxel v clear of
                       (a) other people and (b) static structure?

Occlusion handling comes from the ray marching in visibility_masks(), NOT from
the grid itself. The grid is the substrate.

CONVENTIONS
-----------
world       : metric, X/Y horizontal, Z vertical (up). Ground at Z = 0.
voxel index : (iz, iy, ix) -- NumPy order, so volume[iz] is a horizontal slice
              at height z0 + (iz+0.5)*cell_z. BEV = volume.sum(axis=0).
counts      : every density volume sums to the number of people, exactly as in
              Zhang & Chan (AAAI'20), so volume.sum() is your sanity check.

GROUNDED DEFAULTS
-----------------
cell_xy = 0.5 m   CVCS scene-level density map resolution (CVPR'21 Sec 5.2)
cell_z  = 0.4 m   3D counting voxel height (AAAI'20)
z range = 0..2.8  3D counting height range (AAAI'20) -> 7 levels

For a train compartment these are far too coarse. Use cell_xy = 0.15-0.25 and
cell_z = 0.3; at 0.5 m/cell two standing passengers share a voxel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cvcs_geometry import project_points


# ==========================================================================
# grid
# ==========================================================================
@dataclass
class VoxelGrid:
    x0: float
    y0: float
    z0: float
    cell_xy: float
    cell_z: float
    nx: int
    ny: int
    nz: int

    @classmethod
    def from_points(cls, XYZ: np.ndarray, cell_xy: float = 0.5,
                    cell_z: float = 0.4, margin: float = 5.0,
                    z_min: float = 0.0, z_max: float = 2.8) -> "VoxelGrid":
        """Fit a grid around world points; z range is fixed by human height."""
        XYZ = np.asarray(XYZ, dtype=np.float64).reshape(-1, 3)
        XYZ = XYZ[np.isfinite(XYZ).all(axis=1)]
        mn, mx = XYZ[:, :2].min(0) - margin, XYZ[:, :2].max(0) + margin
        return cls(float(mn[0]), float(mn[1]), float(z_min), cell_xy, cell_z,
                   int(np.ceil((mx[0] - mn[0]) / cell_xy)),
                   int(np.ceil((mx[1] - mn[1]) / cell_xy)),
                   int(np.ceil((z_max - z_min) / cell_z)))

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def n_voxels(self) -> int:
        return self.nz * self.ny * self.nx

    @property
    def voxel_volume(self) -> float:
        return self.cell_xy * self.cell_xy * self.cell_z

    @property
    def bev_extent(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.x0 + self.nx * self.cell_xy,
                self.y0, self.y0 + self.ny * self.cell_xy)

    def centers(self) -> np.ndarray:
        """(n_voxels, 3) world coordinates of voxel centres, in (iz,iy,ix) order."""
        zs = self.z0 + (np.arange(self.nz) + 0.5) * self.cell_z
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.cell_xy
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell_xy
        Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def world_to_index(self, XYZ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(N,3) world -> (indices (N,3) as (iz,iy,ix), inside mask (N,))."""
        XYZ = np.asarray(XYZ, dtype=np.float64).reshape(-1, 3)
        ix = np.floor((XYZ[:, 0] - self.x0) / self.cell_xy)
        iy = np.floor((XYZ[:, 1] - self.y0) / self.cell_xy)
        iz = np.floor((XYZ[:, 2] - self.z0) / self.cell_z)
        idx = np.stack([iz, iy, ix], axis=1)
        inside = (np.isfinite(idx).all(axis=1) &
                  (ix >= 0) & (ix < self.nx) &
                  (iy >= 0) & (iy < self.ny) &
                  (iz >= 0) & (iz < self.nz))
        idx = np.where(np.isfinite(idx), idx, 0).astype(np.int64)
        return idx, inside


# ==========================================================================
# 3D ground-truth density from annotations
# ==========================================================================
def density_volume_3d(grid: VoxelGrid,
                      head_world: np.ndarray,
                      sigma_xy: float = 0.3,
                      sigma_z: float = 0.4,
                      body_from_head: bool = True,
                      person_height: float = 1.75) -> np.ndarray:
    """
    3D scene-level density volume from annotated head world points, using 3D
    Gaussian kernels (Zhang & Chan, AAAI'20). Each person contributes mass
    exactly 1.0, so volume.sum() == number of people inside the grid.

    body_from_head=True spreads each person's mass along a vertical segment
    from the ground to the head instead of putting it all at head height. For
    occlusion work this matters: a person occludes with their whole body, not
    with a blob floating at 1.75 m.
    """
    head_world = np.asarray(head_world, dtype=np.float64).reshape(-1, 3)
    head_world = head_world[np.isfinite(head_world).all(axis=1)]
    vol = np.zeros(grid.shape, dtype=np.float64)
    if head_world.size == 0:
        return vol

    # kernel support in voxels
    rx = max(int(np.ceil(3 * sigma_xy / grid.cell_xy)), 1)
    rz = max(int(np.ceil(3 * sigma_z / grid.cell_z)), 1)

    for p in head_world:
        if body_from_head:
            # sample the body axis; mass split evenly, each sample kernel-normalised
            n_s = max(int(np.ceil(person_height / grid.cell_z)), 1)
            zs = np.linspace(0.05 * person_height, p[2], n_s)
            samples = np.stack([np.full(n_s, p[0]), np.full(n_s, p[1]), zs], axis=1)
            w = 1.0 / n_s
        else:
            samples, w = p.reshape(1, 3), 1.0

        for s in samples:
            iz, iy, ix = grid.world_to_index(s.reshape(1, 3))[0][0]
            z0, z1 = max(iz - rz, 0), min(iz + rz + 1, grid.nz)
            y0, y1 = max(iy - rx, 0), min(iy + rx + 1, grid.ny)
            x0, x1 = max(ix - rx, 0), min(ix + rx + 1, grid.nx)
            if z0 >= z1 or y0 >= y1 or x0 >= x1:
                continue
            zc = grid.z0 + (np.arange(z0, z1) + 0.5) * grid.cell_z
            yc = grid.y0 + (np.arange(y0, y1) + 0.5) * grid.cell_xy
            xc = grid.x0 + (np.arange(x0, x1) + 0.5) * grid.cell_xy
            gz = np.exp(-0.5 * ((zc - s[2]) / sigma_z) ** 2)
            gy = np.exp(-0.5 * ((yc - s[1]) / sigma_xy) ** 2)
            gx = np.exp(-0.5 * ((xc - s[0]) / sigma_xy) ** 2)
            k = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
            ks = k.sum()
            if ks > 0:
                vol[z0:z1, y0:y1, x0:x1] += w * k / ks   # normalised -> mass = w
    return vol


# ==========================================================================
# visibility: frustum + occlusion ray marching
# ==========================================================================
def in_frustum(grid: VoxelGrid, cam, centers: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Boolean (n_voxels,): voxel centre projects inside camera `cam`'s image and
    lies in front of it. This is field-of-view culling only -- no occlusion.
    """
    C = grid.centers() if centers is None else centers
    uv, zc = project_points(cam.K, cam.R, cam.t, C, cam.dist)
    return ((zc > 0) &
            (uv[:, 0] >= 0) & (uv[:, 0] < cam.img_w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < cam.img_h))


def visibility_masks(grid: VoxelGrid,
                     cams: Sequence,
                     occupancy: np.ndarray,
                     static_occluders: Optional[np.ndarray] = None,
                     n_samples: int = 64,
                     tau: float = 1.5,
                     skip_near: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-camera, per-voxel visibility by ray marching from the camera centre to
    each voxel centre through an attenuating medium.

    occupancy        : (nz,ny,nx) non-negative "how much stuff is here". Use the
                       3D density volume, or a binary people volume.
    static_occluders : (nz,ny,nx) optional. Seat backs, partitions, luggage
                       racks, walls. This is the hook for a train compartment;
                       on CVCS pass None.
    tau              : opacity scale, in people per square metre. Occupancy is
                       first converted to people per CUBIC metre (divide by the
                       voxel volume) so the line integral
                           T = integral(density_m3) ds        [people / m^2]
                       is resolution-independent -- change cell size and the
                       occlusion result does not move. Transmittance = exp(-T/tau).
                       tau = 1.5 puts one adult body directly in the line of
                       sight at transmittance ~0.25. Verify on your own data by
                       calling this with a single person and a single camera.
    skip_near        : metres of ray next to the target voxel that are ignored,
                       so a voxel is not counted as occluding itself.

    Returns
        transmittance (n_cams, n_voxels) in [0,1] -- soft visibility
        frustum       (n_cams, n_voxels) bool     -- FOV only
    """
    centers = grid.centers()
    nv = centers.shape[0]
    # people per voxel -> people per cubic metre, so tau has physical meaning
    occ = occupancy.astype(np.float64) / grid.voxel_volume
    if static_occluders is not None:
        occ = occ + static_occluders.astype(np.float64)

    trans = np.zeros((len(cams), nv))
    frust = np.zeros((len(cams), nv), dtype=bool)

    for k, cam in enumerate(cams):
        frust[k] = in_frustum(grid, cam, centers)
        C = cam.center.reshape(1, 3)
        d = centers - C                               # (nv,3) camera -> voxel
        L = np.linalg.norm(d, axis=1, keepdims=True)  # ray lengths
        L = np.maximum(L, 1e-9)

        # sample fractions along each ray, stopping short of the target voxel
        stop = np.clip(1.0 - skip_near / L[:, 0], 0.05, 1.0)
        fr = np.linspace(0.0, 1.0, n_samples)[None, :] * stop[:, None]  # (nv,ns)

        acc = np.zeros(nv)
        step = (L[:, 0] * stop) / n_samples
        for s in range(n_samples):
            P = C + d * fr[:, s][:, None]
            idx, inside = grid.world_to_index(P)
            vals = np.zeros(nv)
            if inside.any():
                iz, iy, ix = idx[inside, 0], idx[inside, 1], idx[inside, 2]
                vals[inside] = occ[iz, iy, ix]
            acc += vals * step
        trans[k] = np.exp(-acc / max(tau, 1e-9))
        trans[k][~frust[k]] = 0.0
    return trans, frust


def coverage_report(grid: VoxelGrid,
                    trans: np.ndarray,
                    frust: np.ndarray,
                    vis_thresh: float = 0.5) -> dict:
    """
    How well is the volume actually observed? This is the number that justifies
    camera placement in a closed space -- and the number a 2D pipeline cannot
    produce at all.
    """
    n_fov = frust.sum(axis=0)                      # cameras with voxel in FOV
    n_vis = (trans > vis_thresh).sum(axis=0)       # cameras with clear LOS
    return {
        "n_cams_in_fov": n_fov.reshape(grid.shape),
        "n_cams_visible": n_vis.reshape(grid.shape),
        "frac_voxels_unseen": float((n_fov == 0).mean()),
        "frac_voxels_occluded_everywhere": float(((n_fov > 0) & (n_vis == 0)).mean()),
        "frac_voxels_multiview": float((n_vis >= 2).mean()),
        "mean_cams_visible": float(n_vis.mean()),
    }


def occlusion_weighted_count(grid: VoxelGrid,
                             density: np.ndarray,
                             trans: np.ndarray,
                             vis_thresh: float = 0.5) -> dict:
    """
    Split the true count into 'people at least one camera can actually see' and
    'people hidden from every camera'. On CVCS you know the truth, so this
    measures how much a visibility-blind pipeline would have missed -- which is
    exactly the quantity your compartment argument rests on.
    """
    n_vis = (trans > vis_thresh).sum(axis=0).reshape(grid.shape)
    total = float(density.sum())
    seen = float(density[n_vis >= 1].sum())
    return {"total": total, "observable": seen, "hidden": total - seen,
            "hidden_fraction": (total - seen) / total if total > 0 else float("nan")}


# ==========================================================================
# annotation-free path: multi-view ray voting
# ==========================================================================
def ray_vote_volume(grid: VoxelGrid,
                    cams: Sequence,
                    head_pixels: Dict[int, np.ndarray],
                    sigma_px: float = 12.0) -> np.ndarray:
    """
    Build an occupancy volume from 2D head detections ONLY -- no world
    coordinates. Each camera casts a ray through every head pixel; voxels near
    many rays from different cameras score high. This is the path that runs on
    a live multi-camera rig.

    head_pixels: {view_id: (N,2) array of (u,v) head points}

    Scoring: for each camera, project all voxel centres, then score each voxel
    by its distance in IMAGE space to the nearest head point of that view
    (Gaussian, sigma_px). Summing across cameras makes real people -- where
    several views agree -- stand out from single-view ray artefacts.

    Note what this does NOT do: it has no depth per detection, so a ray is
    consistent with every voxel along it. Agreement across views is what breaks
    the ambiguity, which is why coverage (above) matters so much in a
    compartment where two cameras may share almost no viewpoint diversity.
    """
    centers = grid.centers()
    score = np.zeros(centers.shape[0])
    for cam in cams:
        pts = head_pixels.get(cam.view_id)
        if pts is None or len(pts) == 0:
            continue
        uv, zc = project_points(cam.K, cam.R, cam.t, centers, cam.dist)
        ok = (zc > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.img_w) \
             & (uv[:, 1] >= 0) & (uv[:, 1] < cam.img_h)
        if not ok.any():
            continue
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        # nearest annotated head in image space, chunked to bound memory
        d2 = np.full(centers.shape[0], np.inf)
        idx = np.flatnonzero(ok)
        for s in range(0, idx.size, 20000):
            sl = idx[s:s + 20000]
            dd = ((uv[sl, None, 0] - pts[None, :, 0]) ** 2 +
                  (uv[sl, None, 1] - pts[None, :, 1]) ** 2)
            d2[sl] = dd.min(axis=1)
        score[ok] += np.exp(-0.5 * d2[ok] / sigma_px ** 2)
    return score.reshape(grid.shape)


# ==========================================================================
# camera selection weights from the CVCS distance map
# ==========================================================================
def camera_weights_from_distance(grid: VoxelGrid,
                                 cams: Sequence,
                                 depth_maps: Sequence[np.ndarray],
                                 eps: float = 1e-6) -> np.ndarray:
    """
    CVCS Eq. (2)-(3) camera selection, applied per voxel: sample each camera's
    stored distance map at the voxel's projected pixel, give the nearest camera
    weight 1 and the others weight exp(-(M_k - M_min)^2), then normalise across
    cameras.

    This is the one place the .h5 distance maps legitimately belong: relative
    camera preference. The weights are ratios of stored values, so they are
    invariant to the unit question -- as long as the stored values are a
    monotone function of distance, which the hypothesis test in cvcs_checks
    confirms. Do not read metric distance off these.
    """
    centers = grid.centers()
    nv = centers.shape[0]
    M = np.full((len(cams), nv), np.inf)
    for k, (cam, dm) in enumerate(zip(cams, depth_maps)):
        uv, zc = project_points(cam.K, cam.R, cam.t, centers, cam.dist)
        ok = (zc > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.img_w) \
             & (uv[:, 1] >= 0) & (uv[:, 1] < cam.img_h)
        if not ok.any():
            continue
        dh, dw = dm.shape
        c = np.clip((uv[ok, 0] * dw / cam.img_w).astype(int), 0, dw - 1)
        r = np.clip((uv[ok, 1] * dh / cam.img_h).astype(int), 0, dh - 1)
        M[k, ok] = dm[r, c]

    M_min = M.min(axis=0)
    valid = np.isfinite(M_min)
    W = np.zeros_like(M)
    with np.errstate(over="ignore", invalid="ignore"):
        W[:, valid] = np.exp(-((M[:, valid] - M_min[valid][None, :]) ** 2))
    W[~np.isfinite(M)] = 0.0
    Z = W.sum(axis=0, keepdims=True)
    return np.divide(W, Z, out=np.zeros_like(W), where=Z > eps)