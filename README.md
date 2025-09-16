
# Deep Learning for Facial 3D Reconstruction Simulator

This project is a deep learning simulator for 3D facial reconstruction. It processes a 3D facial model and generates a diverse set of 2D images, which are then analyzed to find the most frontal view. This process helps standardize models for deep learning pipelines.

---

## 1. 3D Model Normalization

The first stage focuses on preparing the 3D model for consistent use. The `normalize_model` function ensures the model is centered, oriented, and scaled correctly.

* **Centering:** The model's centroid is moved to the world origin (0, 0, 0).
* **Axis Alignment:** Using **Principal Component Analysis (PCA)**, the script reorients the model to a standard **Y-Up, Z-Forward** coordinate system. This is essential for ensuring that the model's "up" and "forward" directions are consistent.
* **Uniform Scaling:** The model is scaled uniformly to fit within a **1x1x1 unit cube**, which standardizes its size for machine learning applications.

---

## 2. Image Generation with Pose Variation

After normalization, a large dataset of 2D images is created by rendering the 3D model from numerous viewpoints.

* A `pyrender` scene is set up with a fixed camera and lighting.
* The script systematically rotates the model using a **3D sweep** of **yaw**, **pitch**, and **roll** angles.
    * **Yaw:** Left-to-right rotation, from -180° to +180°.
    * **Pitch:** Up-and-down rotation, from -60° to +30°.
    * **Roll:** Head tilt, from -90° to +90°.
* For each combination of angles, an image is rendered and saved.

---

## 3. Finding the Most Frontal Image

The final stage uses a deep learning model to analyze the rendered images and identify the most frontal one.

* The `SixDRepNet` model is used to predict the yaw, pitch, and roll angles for each image.
* A **"weighted score"** is calculated for each image. A lower score indicates a more frontal pose, with adjustable weights prioritizing a straight-on view.
* The image with the lowest score is selected as the best frontal image. The angles from this image are then used to reorient the original 3D model, saving the result as `rotated_head.obj`.
* An additional script generates images from different camera heights (chin, nose, eyes) with random jitter and tilt to create an even more realistic and varied dataset.
