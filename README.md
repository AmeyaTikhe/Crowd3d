# Crowd3D: Transparent, Non-DL Multi-Camera Crowd Density Estimation & 3D Reasoning

A modular, interpretable, non-deep-learning multi-camera crowd density estimation and 3D spatial reasoning system built for the **CityStreet** multi-view dataset and designed for extension to **occlusion-aware 3D counting in dense, closed environments (such as train compartments)**.

---

## 1. System Architecture & Directory Structure

The codebase is organized into modular single-responsibility components with a unified CLI pipeline:

```
crowd3d/
├── __init__.py          # Package initialization
├── config.py            # Dataset paths, 3D bounds/resolutions, detection & counting thresholds
├── geometry.py          # Tsai camera calibration, coordinate conversions, VoxelGrid, dataset loader
├── detection.py         # Classical foreground segmentation (median, MOG2), shadow suppression, blobs
├── reconstruction.py    # 3D visual hull carving, multi-view ray voting, ray marching, BEV occupancy
├── counting.py          # 2D Gaussian density map, spatial NMS peak detection, mass-based count
├── evaluation.py        # Hungarian bipartite matching, count metrics (MAE/RMSE), diagnostic dashboard
└── main.py              # End-to-end pipeline runner: Geometry → Detection → Reconstruction → Counting → Evaluation

main.py                  # Workspace root entry point forwarding to crowd3d.main
```

---

## 2. Key Mathematical & Algorithmic Foundations

### A. Camera Model & Coordinate Transformations (`geometry.py`)
- **Tsai Camera Calibration with Radial Distortion**:
  CityStreet views (`view1`, `view2`, `view3`) use Tsai's model with radial distortion parameter $\kappa_1$:
  $$\begin{bmatrix} x_c \\ y_c \\ z_c \end{bmatrix} = R \begin{bmatrix} X_w \\ Y_w \\ -Z_w \end{bmatrix} + t$$
  $$X_u = \frac{x_c}{z_c}, \quad Y_u = \frac{y_c}{z_c}, \quad R_u = \sqrt{X_u^2 + Y_u^2}$$
  $$R_d \cdot (1 + \kappa_1 R_d^2) = R_u \quad \xrightarrow{\text{Newton-Raphson}} \quad R_d$$
  $$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} X_d \cdot s_x / d_{px} + c_x \\ Y_d / d_{py} + c_y \end{bmatrix}$$
- **Exact Bidirectional Ground-Plane Conversion**:
  Relates metric meters $(X_m, Y_m)$ to the standardized CityStreet $640 \times 768$ ground plane:
  $$c_x = \frac{X_m}{0.07625} + 281.6, \quad c_y = \frac{Y_m}{0.07625} + 417.6$$
  $$X_m = (c_x - 281.6) \times 0.07625, \quad Y_m = (c_y - 417.6) \times 0.07625$$

### B. Transparent Non-DL Foreground Extraction (`detection.py`)
- **Temporal Median Background**: Computes static median background across sampled pedestrian frames without requiring manual empty-scene images or neural network training.
- **Photometric Shadow Suppression**: Eliminates cast shadows by detecting luminance drop with chromaticity angle constancy:
  $$\alpha = \frac{I_{gray}}{B_{gray}} \in [0.45, 0.88], \quad \left\|\frac{I}{\|I\|} - \frac{B}{\|B\|}\right\| < \tau_{chroma}$$
- **Morphological Structuring**: Elliptical opening followed by vertical closing to link pedestrian head, torso, and legs.
- **ROI Masking**: Masks out sky and roadway outside the pedestrian monitoring zone.

### C. 3D Voxel Carving & Ray Marching (`reconstruction.py`)
- **Visual Hull Carving**: Evaluates consensus across views. A voxel $V(x, y, z)$ is occupied if:
  $$\sum_{k \in \text{FOV}(V)} \mathbb{I}(\text{mask}_k(\text{proj}_k(V))) \ge N_{min}$$
- **Multi-View Ray Voting**: Accumulates constructive interference Gaussian beams along backprojected sightlines.
- **Volumetric Ray Marching for Closed Spaces**:
  Marches rays from camera optical center $C_k$ to voxel $V_j$ through dynamic human density $\rho_{crowd}$ and static fixture occluders $\rho_{static}$ (train seats, bulkheads):
  $$T_k(V_j) = \exp\left(-\frac{1}{\tau} \int_0^L (\rho_{crowd}(s) + \rho_{static}(s)) \, ds\right)$$

### D. Density Synthesis & Counting (`counting.py`)
- **Density Map**: 2D Gaussian smoothing ($\sigma \approx 0.40\text{ m}$) over projected ground occupancy.
- **Spatial NMS Peak Localization**: Non-maximum suppression enforcing physical human spacing ($d_{min} \approx 0.65\text{ m}$).
- **Mass-Based Integration**: Total density integral normalized by individual footprint area ($M_{person} \approx 1.05\text{ m}^2$).
- **Occlusion Compensation**: Corrects undercounts for commuters hidden behind seats/partitions:
  $$N_{adj} = \sum_{i} \frac{1}{\max(T_{vis}(p_i), \tau_{min})}$$

### E. Evaluation & Metrics (`evaluation.py`)
- **Bipartite Spatial Matching**: Hungarian algorithm matching predicted coordinates to ground truth points at $d_{thresh} = 1.2\text{ m}$.
- Computes: Precision, Recall, F1-Score, Mean Localization Error, MAE, RMSE, and density map correlation.

---

## 3. Usage & CLI Commands

### Run on CityStreet Dataset (Single Frame)
```powershell
python main.py --frame 636 --ray-march
```
Outputs:
- Diagnostic text report: `crowd3d_outputs/report_frame_0636.txt`
- Quantitative metrics JSON: `crowd3d_outputs/metrics_frame_0636.json`
- 6-panel diagnostic dashboard: `crowd3d_outputs/dashboard_frame_0636.png`

### Run on a Multi-Frame Sequence
```powershell
python main.py --frames 636:650:2
```
Outputs individual frame dashboards and sequence summary: `crowd3d_outputs/sequence_summary.json`

### Run Train Compartment 3D Occlusion Simulation
```powershell
python main.py --preset train_compartment
```
Demonstrates how high seat backs, luggage racks, and partition screens occlude seated passengers, and how volumetric ray marching recovers the hidden passenger count.
Output visualization: `crowd3d_compartment_outputs/train_compartment_occlusion_analysis.png`

---

## 4. Key Results

| Metric | CityStreet (Frame 0636) | CityStreet (5-Frame Seq) | Train Compartment |
| :--- | :---: | :---: | :---: |
| **Ground Truth Count** | 82 | 81.4 (avg) | 18 |
| **Predicted Peak Count** | 89 | 73.4 (avg) | 12 (visible) |
| **Occlusion-Adjusted Count** | 89.0 | 73.4 (avg) | **19.8** (recovers hidden) |
| **Absolute Error (AE)** | 7.0 (8.5%) | 12.0 (14.7%) | - |
| **Mean Localization Error** | 0.65 m | 0.61 m | - |
| **Density Map Correlation**| 0.430 | 0.429 | - |
| **FPS / Execution Time** | ~4.4s / frame | ~3.4s / frame | ~0.8s |
