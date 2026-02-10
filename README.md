# Deep Learning for Facial 3D Reconstruction Simulator

This project is a comprehensive tool for processing a 3D facial model, generating a diverse set of 2D images from it, and then using deep learning to find and correct for the most frontal pose. This workflow is particularly useful for preparing 3D assets for machine learning, computer vision, or other applications that require standardised input.
A comprehensive Python toolkit for normalising 3D head models, generating multi-angle rendered images, and performing automated head pose analysis. This pipeline is designed for creating synthetic datasets for head pose estimation, 3D model preprocessing, and computer vision research.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Configuration Reference](#configuration-reference)
- [Output Structure](#output-structure)
- [API Reference](#api-reference)
- [Coordinate System Convention](#coordinate-system-convention)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [Use Cases](#use-cases)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contributing](#contributing)

---

## Overview

This project provides an end-to-end pipeline that:

1. **Normalizes 3D Models** — Centers, reorients, and scales OBJ models to a standard coordinate system
2. **Generates Multi-Angle Renders** — Creates images from systematically varied yaw, pitch, and roll angles
3. **Analyzes Head Pose** — Uses SixDRepNet to identify the most frontal-facing orientation
4. **Produces Final Dataset** — Renders images from multiple camera heights (chin, nose, eye level)

---

## Features

- **Automatic Axis Alignment**: Uses PCA-based analysis to orient models to Y-Up, Z-Forward convention
- **Uniform Scaling**: Fits models within a unit cube while preserving aspect ratio
- **Comprehensive Rotation Sweep**: Generates 1,200+ images covering full 3D rotation space
- **Neural Network Pose Detection**: Leverages SixDRepNet for accurate head pose estimation
- **Multi-Viewpoint Rendering**: Captures models from chin, nose, and eye height perspectives
- **Camera Jitter**: Adds controlled randomness for more realistic training data
- **Green Screen Background**: Easy background removal for downstream processing

---

## Requirements

### System Dependencies

```bash
apt-get update -qq
apt-get install -y xvfb
```

### Python Dependencies

```bash
pip install trimesh[easy]
pip install pyrender
pip install --upgrade pyopengl
pip install scipy
pip install pyvirtualdisplay
pip install pyglet
pip install opencv-python
pip install SixDRepNet
```

---

## Installation

1. **Clone this repository:**
```bash
git clone https://github.com/yourusername/3d-head-pipeline.git
cd 3d-head-pipeline
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
apt-get install -y xvfb
```

3. **(Optional)** For Google Colab, dependencies install automatically via the notebook cells.

---

## Quick Start

### Step 1: Normalize Your 3D Model

```python
from normalize_model import normalize_model

normalize_model(
    input_path='your_model.obj',
    output_path='normalized_model.obj'
)
```

### Step 2: Generate Rotation Sweep Images

```python
# Configure sweep parameters
yaw_steps = 12      # Full 360° horizontal rotation
pitch_steps = 10    # -60° to +30° vertical tilt  
roll_steps = 10     # -180° to +180° head roll

# Generates 1,200 images (12 × 10 × 10)
```

### Step 3: Find Most Frontal Pose

```python
from find_frontal import find_most_frontal_image

best_image, angles = find_most_frontal_image(
    image_folder='rendered_images',
    yaw_weight=2.5,
    pitch_weight=1.5,
    roll_weight=0.5
)
```

### Step 4: Generate Final Dataset

```python
# Renders from three camera heights:
# - Chin level (y = -1.0)
# - Nose level (y = -0.6)  
# - Eye level  (y = -0.2)
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Raw 3D Model (.obj)                   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: Normalization                       │
│  • Center at origin                                             │
│  • PCA-based axis alignment                                     │
│  • Uniform scaling to unit cube                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: Axis Verification                   │
│  • Heuristic check for Y-Up orientation                         │
│  • Corrective rotation if needed                                │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STAGE 3: Full Rotation Sweep                    │
│  • 12 yaw × 10 pitch × 10 roll = 1,200 images                   │
│  • Systematic coverage of 3D rotation space                     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               STAGE 4: Frontal Pose Detection                   │
│  • SixDRepNet neural network analysis                           │
│  • Weighted scoring: yaw > pitch > roll                         │
│  • Constraint filtering for valid poses                         │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 5: Model Pose Correction                     │
│  • Apply inverse rotation to align frontal                      │
│  • Export corrected model                                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│            STAGE 6: Multi-Viewpoint Rendering                   │
│  • Chin height camera (looking up)                              │
│  • Nose height camera (eye level)                               │
│  • Eye height camera (slightly above)                           │
│  • Camera jitter for variation                                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              OUTPUT: Rendered Image Dataset                     │
│  • 30 final images (10 per viewpoint)                           │
│  • Green screen background                                      │
│  • 2048×2048 resolution                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration Reference

### Normalization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tolerance` | 1e-4 | Acceptable centroid offset after centering |

### Rotation Sweep Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `yaw_steps` | 12 | Divisions for Y-axis rotation (-180° to +180°) |
| `pitch_steps` | 10 | Divisions for X-axis rotation (-60° to +30°) |
| `roll_steps` | 10 | Divisions for Z-axis rotation (-180° to +180°) |

### Pose Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yaw_weight` | 2.5 | Importance of horizontal facing direction |
| `pitch_weight` | 1.5 | Importance of vertical head tilt |
| `roll_weight` | 0.5 | Importance of head roll (least significant) |
| `pitch_min/max` | 0°/15° | Acceptable pitch range |
| `roll_min/max` | 0°/15° | Acceptable roll range |
| `yaw_min/max` | -25°/25° | Acceptable yaw range |

### Rendering Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_resolution` | (2048, 2048) | Output image dimensions |
| `jitter_strength` | 0.05 | Camera position randomization |
| `tilt_strength_deg` | 2.0 | Camera rotation randomization |
| `light_intensity` | 40.0 | Directional light brightness |
| `ambient_light` | 0.7 | Ambient illumination level |
| `fov` | π/3 | Camera field of view (60°) |

### Viewpoint Configurations

| Viewpoint | Y Position | Distance | Description |
|-----------|------------|----------|-------------|
| `chin_height` | -1.0 | 2.0 | Camera below chin, looking up |
| `nose_height` | -0.6 | 2.0 | Camera at nose level |
| `eyes_height` | -0.2 | 2.0 | Camera at eye level |

---

## Output Structure

```
rendered_images/
├── head_yaw00_pitch00_roll00.png
├── head_yaw00_pitch00_roll01.png
├── ...
└── head_yaw11_pitch09_roll09.png

most_frontal_image.png          # Best frontal pose with axis visualization
rotated_head.obj                # Pose-corrected 3D model

rendered_images_final/
├── head_chin_height_00.png
├── head_chin_height_01.png
├── ...
├── head_nose_height_00.png
├── ...
├── head_eyes_height_00.png
└── ...
```

---

## API Reference

### `normalize_model(input_path, output_path)`

Normalizes a 3D model by centering, reorienting, and scaling.

**Parameters:**
- `input_path` (str): Path to input OBJ file
- `output_path` (str): Path for normalized output

**Process:**
1. Centers mesh at world origin
2. Computes principal inertia vectors via PCA
3. Disambiguates axis directions using geometry
4. Constructs orthonormal basis and rotation matrix
5. Scales uniformly to fit in unit cube

---

### `generate_full_3d_rotations(yaw_steps, pitch_steps, roll_steps)`

Generates arrays of rotation angles for systematic pose sampling.

**Parameters:**
- `yaw_steps` (int): Number of divisions for yaw (left-right rotation)
- `pitch_steps` (int): Number of divisions for pitch (up-down rotation)
- `roll_steps` (int): Number of divisions for roll (head tilt)

**Returns:**
- `yaw_angles`: Array of Y-axis rotations (radians)
- `pitch_angles`: Array of X-axis rotations (radians)
- `roll_angles`: Array of Z-axis rotations (radians)

---

### `lookAt(eye, target, up)`

Computes a camera pose matrix for positioning and orienting the camera.

**Parameters:**
- `eye` (np.array): Camera position in world coordinates
- `target` (np.array): Point the camera looks at
- `up` (np.array): Up direction vector (default: Y-up)

**Returns:**
- 4×4 transformation matrix

---

## Coordinate System Convention

This pipeline uses the following coordinate system:

```
        +Y (Up)
         │
         │
         │
         └───────── +X (Right)
        /
       /
      /
    +Z (Forward/Camera)
```

- **Y-Axis**: Points upward (top of head)
- **Z-Axis**: Points forward (direction face is looking)
- **X-Axis**: Points to the right (right ear direction)

---

## Troubleshooting

### Common Issues

**1. "Virtual display not started"**
- This is normal on desktop environments
- Ensure `xvfb` is installed for headless servers

**2. "PyRender requires PyOpenGL==3.1.0" warning**
- The upgraded PyOpenGL (3.1.10) works fine
- This warning can be safely ignored

**3. Model appears rotated incorrectly**
- The PCA-based alignment depends on mesh geometry
- For models with shoulders, vertices may bias the principal axes
- Adjust the `CAMERA_TARGET_OFFSET_Y` parameter

**4. SixDRepNet not detecting faces**
- Ensure rendered images show sufficient facial features
- Adjust lighting intensity if images are too dark/bright
- Verify pitch/roll constraints aren't too restrictive

### Performance Tips

- Reduce `image_resolution` for faster processing
- Decrease rotation step counts for quick tests
- Use GPU acceleration with `gpu_id=0` for SixDRepNet

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| trimesh | ≥4.7.4 | 3D mesh loading and manipulation |
| pyrender | ≥0.1.45 | OpenGL-based rendering |
| numpy | ≥1.19.5 | Numerical computations |
| opencv-python | ≥4.5.5 | Image processing |
| SixDRepNet | ≥0.1.6 | Head pose estimation |
| scipy | ≥1.5.4 | Scientific computing |
| Pillow | ≥8.4.0 | Image file handling |
| pyvirtualdisplay | latest | Headless display support |

---

## Use Cases

- **Synthetic Training Data**: Generate labeled head pose datasets
- **3D Model Preprocessing**: Standardize models from various sources
- **Head Pose Estimation Research**: Benchmark pose detection algorithms
- **AR/VR Applications**: Prepare head models for real-time rendering
- **Animation Pipelines**: Orient character heads consistently

---

## License

This project is provided for educational and research purposes.

---

## Acknowledgments

- [SixDRepNet](https://github.com/thohemp/SixDRepNet) for head pose estimation
- [Trimesh](https://trimsh.org/) for 3D mesh processing
- [PyRender](https://pyrender.readthedocs.io/) for 3D rendering

---

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
