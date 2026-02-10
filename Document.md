# Technical Documentation: 3D Head Model Processing Pipeline

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Module Reference](#3-module-reference)
4. [Algorithm Details](#4-algorithm-details)
5. [Configuration Guide](#5-configuration-guide)
6. [Usage Examples](#6-usage-examples)
7. [Extending the Pipeline](#7-extending-the-pipeline)
8. [Troubleshooting Guide](#8-troubleshooting-guide)

---

## 1. Introduction

### 1.1 Purpose

This pipeline provides a complete solution for processing 3D head models from raw OBJ files to rendered image datasets. It addresses common challenges in 3D model preprocessing:

- **Inconsistent orientations**: Models from different sources use different coordinate systems
- **Variable scaling**: Models may have arbitrary scales
- **Unknown pose**: The "frontal" direction may not be aligned with any axis

### 1.2 Target Audience

- Computer vision researchers creating synthetic training data
- 3D artists needing batch model preprocessing
- Developers building head pose estimation systems
- Anyone working with 3D head scans or models

### 1.3 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8+ | 3.10+ |
| RAM | 4 GB | 8 GB |
| GPU | Not required | CUDA-capable (for SixDRepNet) |
| OS | Linux, macOS, Windows | Linux (Ubuntu 20.04+) |

---

## 2. System Architecture

### 2.1 Pipeline Overview

The pipeline consists of six sequential stages, each building upon the previous:

```
┌────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: NORMALIZATION                     │
│  Input:  Raw .obj file                                             │
│  Output: normalized_model.obj                                      │
│  Key Functions: normalize_model()                                  │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                      STAGE 2: AXIS VERIFICATION                    │
│  Input:  normalized_model.obj                                      │
│  Output: reoriented_model.obj                                      │
│  Key Functions: heuristic axis check, rotation correction          │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                     STAGE 3: ROTATION SWEEP                        │
│  Input:  reoriented_model.obj                                      │
│  Output: 1,200 rendered images                                     │
│  Key Functions: generate_full_3d_rotations(), pyrender rendering   │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                   STAGE 4: FRONTAL DETECTION                       │
│  Input:  Rendered images                                           │
│  Output: Best frontal image, pose angles                           │
│  Key Functions: SixDRepNet.predict(), weighted scoring             │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                    STAGE 5: POSE CORRECTION                        │
│  Input:  Original model + detected angles                          │
│  Output: rotated_head.obj                                          │
│  Key Functions: euler_matrix(), mesh.apply_transform()             │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│                  STAGE 6: MULTI-VIEW RENDERING                     │
│  Input:  rotated_head.obj                                          │
│  Output: 30 final rendered images                                  │
│  Key Functions: make_images(), camera jitter                       │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Raw Model (.obj)
     │
     ├─► Vertex Data ─────────► Centroid Calculation
     │                                    │
     │                                    ▼
     │                          Translation Transform
     │                                    │
     ├─► Mesh Geometry ──────► PCA Analysis (Principal Axes)
     │                                    │
     │                                    ▼
     │                          Rotation Matrix Construction
     │                                    │
     ├─► Bounding Box ────────► Scale Factor Calculation
     │                                    │
     │                                    ▼
     │                          Combined Transform Matrix
     │                                    │
     └──────────────────────────────────► Normalized Model
                                              │
                                              ▼
                                    Rendered Image Grid
                                              │
                                              ▼
                                    Neural Pose Analysis
                                              │
                                              ▼
                                    Final Rotated Model
                                              │
                                              ▼
                                    Multi-View Dataset
```

---

## 3. Module Reference

### 3.1 Model Normalization Module

#### `normalize_model(input_path: str, output_path: str) -> None`

Loads, normalizes, and exports a 3D model.

**Algorithm:**

1. Load mesh using `trimesh.load(force='mesh')`
2. Flatten scene if multiple meshes present
3. Center: `mesh.apply_translation(-mesh.centroid)`
4. Get PCA vectors: `mesh.principal_inertia_vectors`
5. Disambiguate axis directions using geometry
6. Construct rotation matrix from orthonormal basis
7. Scale: `mesh.apply_scale(1.0 / max_extent)`
8. Export to output path

**Code Snippet:**

```python
def normalize_model(input_path: str, output_path: str):
    mesh = trimesh.load(input_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    # Center at origin
    mesh.apply_translation(-mesh.centroid)
    
    # Get principal axes
    pca_vectors = np.copy(mesh.principal_inertia_vectors)
    source_up_down = pca_vectors[0]
    source_front_back = pca_vectors[2]
    
    # Disambiguate directions using geometry
    geometric_up = mesh.vertices[np.argmax(mesh.vertices[:, 1])] - \
                   mesh.vertices[np.argmin(mesh.vertices[:, 1])]
    if np.dot(source_up_down, geometric_up) < 0:
        source_up_down *= -1
    
    # Build orthonormal basis and apply rotation
    final_up = source_up_down / np.linalg.norm(source_up_down)
    final_forward = source_front_back - np.dot(source_front_back, final_up) * final_up
    final_forward /= np.linalg.norm(final_forward)
    final_right = np.cross(final_up, final_forward)
    
    source_basis = np.stack([final_right, final_up, final_forward], axis=1)
    rotation_matrix = np.eye(3) @ source_basis.T
    
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    mesh.apply_transform(transform)
    
    # Uniform scaling
    max_extent = np.max(mesh.extents)
    mesh.apply_scale(1.0 / max_extent)
    
    mesh.export(output_path)
```

### 3.2 Camera Utilities Module

#### `lookAt(eye, target, up) -> np.ndarray`

Computes a 4×4 camera pose matrix.

**Parameters:**
| Name | Type | Default | Description |
|------|------|---------|-------------|
| `eye` | np.array(3) | required | Camera position |
| `target` | np.array(3) | required | Look-at point |
| `up` | np.array(3) | [0,1,0] | Up direction |

**Returns:** 4×4 transformation matrix

**Mathematical Formulation:**

```
z_axis = normalize(eye - target)     # Forward vector
x_axis = normalize(up × z_axis)      # Right vector  
y_axis = z_axis × x_axis             # True up vector

        ┌                      ┐
        │ x_x  y_x  z_x  eye_x │
pose =  │ x_y  y_y  z_y  eye_y │
        │ x_z  y_z  z_z  eye_z │
        │  0    0    0     1   │
        └                      ┘
```

### 3.3 Rotation Generation Module

#### `generate_full_3d_rotations(yaw_steps, pitch_steps, roll_steps)`

Generates systematic rotation angles for 3D pose sampling.

**Angle Ranges:**

| Angle | Range (radians) | Range (degrees) | Description |
|-------|-----------------|-----------------|-------------|
| Yaw | -π to +π | -180° to +180° | Full horizontal sweep |
| Pitch | -π/3 to +π/6 | -60° to +30° | Looking up to slightly down |
| Roll | -π to +π | -180° to +180° | Full head tilt range |

**Returns:** Tuple of three numpy arrays (yaw_angles, pitch_angles, roll_angles)

### 3.4 Scene Setup Module

#### `load_and_normalize_model(scene, file_path)`

Loads a mesh into a PyRender scene with normalization.

**Process:**
1. Load with `trimesh.load(force='mesh')`
2. Concatenate if scene
3. Fix normals
4. Center and scale to unit cube
5. Create PyRender mesh with smooth shading
6. Add to scene

#### `setup_fixed_scene(scene)`

Configures camera and lighting for consistent rendering.

**Settings:**
- Camera position: (0, 0, 1.2)
- Camera FOV: 60° (π/3 radians)
- Aspect ratio: 1:1
- Directional light intensity: 40.0
- Ambient light: 0.7 (all channels)

#### `create_light(scene, light_pose, light_intensity=15.0)`

Adds a directional white light to the scene.

### 3.5 Pose Detection Module

Uses SixDRepNet for neural network-based head pose estimation.

#### Weighted Scoring Formula

```
score = √(w_pitch × pitch² + w_yaw × yaw² + w_roll × roll²)
```

**Default Weights:**
| Weight | Value | Rationale |
|--------|-------|-----------|
| `yaw_weight` | 2.5 | Most important: Is face toward camera? |
| `pitch_weight` | 1.5 | Moderately important: Head tilt up/down |
| `roll_weight` | 0.5 | Least important: Slight tilt acceptable |

#### Constraint Filters

| Constraint | Min | Max | Purpose |
|------------|-----|-----|---------|
| Pitch | 0° | 15° | Reject extreme up/down |
| Roll | 0° | 15° | Reject excessive tilt |
| Yaw | -25° | 25° | Allow some left/right |

### 3.6 Final Rendering Module

#### `make_images(scene, model_node, renderer, ...)`

Generates images with camera jitter for dataset variation.

**Jitter Application:**
1. Start with base camera position
2. Add uniform random offset (±jitter_strength)
3. Apply random roll (±tilt_strength_deg)
4. Apply random pitch (±tilt_strength_deg)
5. Recompute lookAt matrix

---

## 4. Algorithm Details

### 4.1 PCA-Based Axis Alignment

The Principal Component Analysis alignment works as follows:

**Step 1: Compute Principal Inertia Vectors**
```python
pca_vectors = mesh.principal_inertia_vectors
```

This returns three orthogonal vectors representing the principal axes of the mesh's inertia tensor, ordered by eigenvalue magnitude.

**Step 2: Assign Semantic Meaning**
- Vector 0 (largest variance): Up/Down axis
- Vector 1 (medium variance): Left/Right axis  
- Vector 2 (smallest variance): Front/Back axis

**Step 3: Disambiguate Directions**

PCA vectors are bidirectional. We use geometry to determine correct orientation:

```python
# Y-axis (up) disambiguation
geometric_up = mesh.vertices[np.argmax(mesh.vertices[:, 1])] - \
               mesh.vertices[np.argmin(mesh.vertices[:, 1])]
if np.dot(source_up_down, geometric_up) < 0:
    source_up_down *= -1

# Z-axis (forward) disambiguation
projections = np.dot(mesh.vertices, source_front_back)
nose_point = mesh.vertices[np.argmax(projections)]
if np.dot(nose_point, source_front_back) < 0:
    source_front_back *= -1
```

**Step 4: Construct Orthonormal Basis**

Gram-Schmidt orthogonalization ensures perpendicular axes:

```python
final_up = normalize(source_up_down)
final_forward = source_front_back - dot(source_front_back, final_up) * final_up
final_forward = normalize(final_forward)
final_right = cross(final_up, final_forward)
```

### 4.2 Euler Angle Rotation Composition

Rotations are applied in the order: Yaw → Pitch → Roll

```python
Ry = rotation_matrix(yaw, [0, 1, 0])    # Y-axis rotation
Rx = rotation_matrix(pitch, [1, 0, 0])   # X-axis rotation
Rz = rotation_matrix(roll, [0, 0, 1])    # Z-axis rotation

pose = Ry @ Rx @ Rz  # Matrix multiplication order
```

This follows the **extrinsic** rotation convention where each rotation is about the fixed world axes.

### 4.3 Head Pose Estimation with SixDRepNet

SixDRepNet predicts head pose using 6D rotation representation:

1. **Input**: RGB image containing a face
2. **Output**: Three Euler angles (pitch, yaw, roll) in degrees
3. **Architecture**: ResNet-50 backbone with 6D rotation head

**Advantages over direct angle regression:**
- Continuous representation avoids gimbal lock
- More stable gradients during training
- Better generalization to extreme poses

### 4.4 Weighted Frontal Score

The scoring function prioritizes angles differently:

```python
score = np.sqrt(
    pitch_weight * pitch**2 + 
    yaw_weight * yaw**2 + 
    roll_weight * roll**2
)
```

**Interpretation:**
- Lower score = more frontal
- Yaw (left/right facing) weighted highest because it most affects "facing camera"
- Pitch (up/down tilt) weighted medium
- Roll (head tilt) weighted lowest as slight roll is natural

---

## 5. Configuration Guide

### 5.1 Normalization Settings

```python
# In normalize_model function
tolerance = 1e-4  # Centroid offset tolerance
```

### 5.2 Rotation Sweep Settings

```python
# Main script configuration
yaw_steps = 12    # 360° ÷ 12 = 30° increments
pitch_steps = 10  # 90° ÷ 10 = 9° increments
roll_steps = 10   # 360° ÷ 10 = 36° increments

# Total images: 12 × 10 × 10 = 1,200
```

**Customization Examples:**

```python
# Quick test (64 images)
yaw_steps, pitch_steps, roll_steps = 4, 4, 4

# High density (10,000 images)
yaw_steps, pitch_steps, roll_steps = 20, 20, 25
```

### 5.3 Pose Detection Settings

```python
# Scoring weights
yaw_weight = 2.5    # Range: 1.0 - 5.0
pitch_weight = 1.5  # Range: 1.0 - 3.0
roll_weight = 0.5   # Range: 0.1 - 1.0

# Constraint filters
pitch_min, pitch_max = 0, 15    # Degrees
roll_min, roll_max = 0, 15      # Degrees
yaw_min, yaw_max = -25, 25      # Degrees
```

**Tuning Guidelines:**
- Increase `yaw_weight` if frontal alignment is poor
- Widen `yaw_min/max` if no images pass filtering
- Decrease `pitch_weight` if slight tilts are acceptable

### 5.4 Rendering Settings

```python
# Resolution
image_resolution = (2048, 2048)  # (width, height)

# Camera settings
camera_fov = np.pi / 3.0  # 60 degrees
camera_distance = 2.0
aspect_ratio = 1.0

# Lighting
light_intensity = 40.0
ambient_light = 0.7

# Background (RGBA)
bg_color = np.array([0, 1.0, 0, 1.0])  # Green screen

# Jitter
jitter_strength = 0.05      # Position offset
tilt_strength_deg = 2.0     # Rotation offset
```

### 5.5 Viewpoint Configurations

```python
viewpoints = [
    {'name': 'chin_height',  'y_pos': -1.0, 'dist': 2.0},
    {'name': 'nose_height',  'y_pos': -0.6, 'dist': 2.0},
    {'name': 'eyes_height',  'y_pos': -0.2, 'dist': 2.0},
]

# Camera target offset (for models with shoulders)
CAMERA_TARGET_OFFSET_Y = 0.15
```

---

## 6. Usage Examples

### 6.1 Basic Usage: Process Single Model

```python
import numpy as np
import trimesh
import pyrender
from PIL import Image

# Step 1: Normalize
normalize_model('raw_head.obj', 'normalized.obj')

# Step 2: Render sweep
scene = pyrender.Scene()
model_node = load_and_normalize_model(scene, 'normalized.obj')
setup_fixed_scene(scene)

renderer = pyrender.OffscreenRenderer(512, 512)
yaw_angles, pitch_angles, roll_angles = generate_full_3d_rotations(8, 6, 6)

for y_idx, yaw in enumerate(yaw_angles):
    for p_idx, pitch in enumerate(pitch_angles):
        for r_idx, roll in enumerate(roll_angles):
            pose = Ry(yaw) @ Rx(pitch) @ Rz(roll)
            scene.set_pose(model_node, pose)
            color, _ = renderer.render(scene)
            Image.fromarray(color).save(f'render_{y_idx}_{p_idx}_{r_idx}.png')
```

### 6.2 Batch Processing Multiple Models

```python
import os
from pathlib import Path

input_dir = Path('raw_models/')
output_dir = Path('processed/')

for obj_file in input_dir.glob('*.obj'):
    # Create output subdirectory
    model_output = output_dir / obj_file.stem
    model_output.mkdir(exist_ok=True)
    
    # Normalize
    normalized_path = model_output / 'normalized.obj'
    normalize_model(str(obj_file), str(normalized_path))
    
    # Render
    render_output = model_output / 'renders'
    render_output.mkdir(exist_ok=True)
    
    # ... rendering code ...
    
    print(f"Processed: {obj_file.name}")
```

### 6.3 Custom Viewpoint Configuration

```python
custom_viewpoints = [
    {'name': 'overhead',     'y_pos': 1.0,  'dist': 2.5},
    {'name': 'eye_level',    'y_pos': 0.0,  'dist': 1.5},
    {'name': 'low_angle',    'y_pos': -1.5, 'dist': 2.5},
    {'name': 'profile_left', 'y_pos': 0.0,  'dist': 1.5},
]

for view in custom_viewpoints:
    scene = pyrender.Scene(bg_color=np.array([0, 0, 0, 1]))  # Black background
    model_node = load_and_normalize_model(scene, 'model.obj')
    
    eye_pos = np.array([0.0, view['y_pos'], view['dist']])
    setup_scene_and_camera(scene, eye_position=eye_pos)
    
    # Render...
```

### 6.4 Custom Pose Detection Weights

```python
from sixdrepnet import SixDRepNet

model = SixDRepNet(gpu_id=-1)

# Strict frontal detection (only nearly perfect alignment)
strict_weights = {
    'yaw_weight': 5.0,
    'pitch_weight': 3.0,
    'roll_weight': 2.0
}

# Relaxed detection (allow more variation)
relaxed_weights = {
    'yaw_weight': 1.5,
    'pitch_weight': 1.0,
    'roll_weight': 0.5
}

def compute_score(pitch, yaw, roll, weights):
    return np.sqrt(
        weights['pitch_weight'] * pitch**2 +
        weights['yaw_weight'] * yaw**2 +
        weights['roll_weight'] * roll**2
    )
```

---

## 7. Extending the Pipeline

### 7.1 Adding New Renderers

```python
class CustomRenderer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def render(self, scene):
        # Custom rendering logic
        color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        return color, depth
    
    def delete(self):
        pass

# Usage
renderer = CustomRenderer(512, 512)
color, depth = renderer.render(scene)
```

### 7.2 Adding Custom Pose Estimators

```python
class CustomPoseEstimator:
    def __init__(self):
        # Initialize your model
        pass
    
    def predict(self, image):
        """
        Returns: (pitch, yaw, roll) in degrees
        """
        # Your inference code
        return pitch, yaw, roll

# Integration
estimator = CustomPoseEstimator()

for img_path in image_paths:
    img = cv2.imread(img_path)
    pitch, yaw, roll = estimator.predict(img)
    # Process...
```

### 7.3 Adding New Normalization Methods

```python
def normalize_with_landmarks(mesh, landmarks):
    """
    Normalize using facial landmarks instead of PCA.
    
    landmarks: dict with keys 'left_eye', 'right_eye', 'nose', 'chin'
    """
    # Compute forward vector from nose
    forward = landmarks['nose'] - mesh.centroid
    forward /= np.linalg.norm(forward)
    
    # Compute up vector from eyes
    eye_center = (landmarks['left_eye'] + landmarks['right_eye']) / 2
    up = eye_center - landmarks['chin']
    up /= np.linalg.norm(up)
    
    # Orthogonalize
    up = up - np.dot(up, forward) * forward
    up /= np.linalg.norm(up)
    
    right = np.cross(up, forward)
    
    # Build rotation matrix
    rotation = np.stack([right, up, forward], axis=1)
    
    transform = np.eye(4)
    transform[:3, :3] = rotation.T
    mesh.apply_transform(transform)
    
    return mesh
```

---

## 8. Troubleshooting Guide

### 8.1 Installation Issues

**Problem:** `PyOpenGL` version conflict
```
ERROR: pyrender 0.1.45 requires PyOpenGL==3.1.0
```

**Solution:** The newer PyOpenGL (3.1.10) is compatible. Ignore the warning.

**Problem:** `Xvfb` not found
```
xvfb-run: command not found
```

**Solution:**
```bash
apt-get update && apt-get install -y xvfb
```

### 8.2 Rendering Issues

**Problem:** Black/empty renders

**Causes & Solutions:**
1. Model not in view → Adjust camera distance
2. Lighting too low → Increase `light_intensity`
3. Model scale wrong → Verify normalization

**Problem:** Model appears rotated incorrectly

**Causes & Solutions:**
1. PCA misalignment → Check vertex distribution
2. Shoulders biasing axis → Adjust `CAMERA_TARGET_OFFSET_Y`
3. Non-standard OBJ format → Verify OBJ coordinate system

### 8.3 Pose Detection Issues

**Problem:** No images pass constraints

**Solutions:**
1. Widen constraint ranges:
```python
pitch_min, pitch_max = -30, 30
roll_min, roll_max = -30, 30
yaw_min, yaw_max = -45, 45
```

2. Check if face is visible in renders
3. Verify SixDRepNet installation

**Problem:** Wrong image selected as most frontal

**Solutions:**
1. Adjust weights:
```python
yaw_weight = 3.0   # Increase yaw importance
pitch_weight = 1.0
roll_weight = 0.3  # Decrease roll importance
```

2. Check render lighting/visibility

### 8.4 Performance Issues

**Problem:** Rendering too slow

**Solutions:**
1. Reduce resolution:
```python
image_resolution = (512, 512)
```

2. Reduce rotation steps:
```python
yaw_steps, pitch_steps, roll_steps = 6, 5, 5  # 150 images
```

3. Enable GPU for SixDRepNet:
```python
model = SixDRepNet(gpu_id=0)
```

### 8.5 Memory Issues

**Problem:** Out of memory during batch rendering

**Solutions:**
1. Process in smaller batches
2. Delete renderer between batches:
```python
renderer.delete()
renderer = pyrender.OffscreenRenderer(width, height)
```

3. Clear scene objects:
```python
scene.clear()
```

---

## Appendix A: Coordinate System Reference

### Standard Orientation (Y-Up, Z-Forward)

```
                    +Y (Up)
                     │
                     │    
                     │   
         ─ ─ ─ ─ ─ ─ ┼ ─ ─ ─ ─ ─ ─ +X (Right)
                    /│
                   / │
                  /  │
               +Z    │
            (Forward)│
                     │

        Face Direction: +Z
        Top of Head:   +Y
        Right Ear:     +X
```

### Rotation Axis Definitions

| Rotation | Axis | Positive Direction | Motion |
|----------|------|-------------------|--------|
| Yaw | Y | Counter-clockwise from above | Turn left |
| Pitch | X | Counter-clockwise from right | Look up |
| Roll | Z | Counter-clockwise from front | Tilt left |

---

## Appendix B: File Format Reference

### Input OBJ Requirements

- Vertex positions (`v x y z`)
- Face definitions (`f v1 v2 v3`)
- Optional: Normals, texture coordinates

### Output File Structure

```
project/
├── raw_model.obj              # Input
├── normalized_model.obj       # After normalization
├── reoriented_model.obj       # After axis verification
├── rotated_head.obj           # After pose correction
├── rendered_images/           # Rotation sweep
│   ├── head_yaw00_pitch00_roll00.png
│   └── ...
├── most_frontal_image.png     # Best frontal with axes
└── final_renders/             # Multi-viewpoint output
    ├── head_chin_height_00.png
    ├── head_nose_height_00.png
    └── head_eyes_height_00.png
```

---

## Appendix C: Dependencies Version Matrix

| Package | Tested Version | Minimum Version |
|---------|---------------|-----------------|
| trimesh | 4.7.4 | 3.9.0 |
| pyrender | 0.1.45 | 0.1.40 |
| numpy | 2.0.2 | 1.19.0 |
| opencv-python | 4.12.0.88 | 4.5.0 |
| SixDRepNet | 0.1.6 | 0.1.0 |
| scipy | 1.16.1 | 1.5.0 |
| Pillow | 11.3.0 | 8.0.0 |
| torch | 2.8.0 | 1.10.0 |
| pyvirtualdisplay | latest | 2.0 |

---

*Documentation Version: 1.0*  
*Last Updated: February 2026*
