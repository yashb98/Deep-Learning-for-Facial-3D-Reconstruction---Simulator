# Deep Learning for Facial 3D Reconstruction Simulator

This project is a comprehensive tool for processing a 3D facial model, generating a diverse set of 2D images from it, and then using deep learning to find and correct for the most frontal pose. This workflow is particularly useful for preparing 3D assets for machine learning, computer vision, or other applications that require standardized input.

## Features

  * **3D Model Normalization:** Automatically centers, orients, and scales 3D models for consistent use across different datasets.
  * **Robust Axis Alignment:** Uses Principal Component Analysis (PCA) and geometric heuristics to align models to a standard Y-Up, Z-Forward coordinate system.
  * **Customizable 3D Pose Generation:** Renders a large number of images by systematically rotating the model across yaw, pitch, and roll angles.
  * **Head Pose Estimation:** Employs the `SixDRepNet` deep learning model to accurately predict the head pose (yaw, pitch, roll) from a 2D image.
  * **Frontal Pose Correction:** Identifies the most frontal image from the generated dataset using a weighted scoring system and applies the corresponding rotation to the original 3D model.
  * **Varied Viewpoint Generation:** Includes a separate script to generate images from multiple camera heights (chin, nose, eyes) with random jitter and tilt to create more realistic and varied datasets.

## Installation

This project is designed to run in a Google Colab environment, which simplifies the setup. The required libraries are installed at the beginning of the notebook. If you are running this locally, you will need to install the dependencies manually.

1.  **Install Required Libraries:**
    ```bash
    !pip install trimesh[easy]
    !pip install pyrender
    !pip install --upgrade pyopengl
    !pip install scipy
    !pip install pyvirtualdisplay -q
    !pip install pyglet
    !pip install opencv-python
    !pip install SixDRepNet
    !apt-get update -qq
    !apt-get install -y xvfb -qq
    ```
      * `Trimesh` is used for 3D mesh processing.
      * `PyRender` is for 3D rendering.
      * `PyOpenGL` is a dependency for `PyRender`.
      * `PyVirtualDisplay` and `Xvfb` are for headless rendering in environments like Google Colab.
      * `SixDRepNet` is a library for 6D head pose estimation.

## How to Use

The workflow is broken down into three main code blocks within the notebook.

### Step 1: Normalize the 3D Model

This script centers, orients, and scales your 3D model.

1.  **Place your `.obj` file** in the same directory as the notebook.
2.  **Update the `INPUT_FILE` variable** in the main block of the `normalize_model` function to point to your model's file path.
    ```python
    INPUT_FILE = '/content/your_model.obj'
    OUTPUT_FILE = 'your_model_normalized.obj'
    ```
3.  **Run the cell.** The script will output a new file named `your_model_normalized.obj`.

The `normalize_model` function performs the following transformations:

  * **Centering:** Moves the mesh so its centroid is at the origin.
  * **Axis Alignment:** Uses PCA to find the principal axes and rotates the model to a standard Y-Up, Z-Forward orientation.
  * **Uniform Scaling:** Scales the model so its largest dimension is 1.0, ensuring it fits within a unit cube.

### Step 2: Generate 2D Images from Multiple Poses

This section renders a large number of images from various head poses (yaw, pitch, and roll).

1.  **Update the `model_filename` variable** to the output of the previous step:
    ```python
    model_filename = "your_model_normalized.obj"
    ```
2.  **Configure the `yaw_steps`, `pitch_steps`, and `roll_steps`** to control the number of images to generate. More steps result in a more detailed sweep but take longer to process.
3.  **Run the cell.** The script will create a new directory (`rendered_images` by default) and save the generated images in `.png` format.

### Step 3: Find the Most Frontal Image and Correct the 3D Model

This final step analyzes the generated images to find the one closest to a frontal pose and then applies that correction to the 3D model.

1.  **Update the paths** to match your output directory and model files:
    ```python
    image_folder = '/content/rendered_images'
    output_path = '/content/most_frontal_image.png'
    model_path = '/content/your_model_normalized.obj'
    rotated_model_path = '/content/rotated_head.obj'
    ```
2.  **Adjust the `yaw_weight`, `pitch_weight`, and `roll_weight`** variables to fine-tune the frontal pose criteria. A higher weight makes that angle's deviation from zero more penalized in the final score.
3.  **Run the cell.** The script will iterate through the images, calculate a weighted score for each, and identify the one with the lowest score. It will then:
      * Print the filename and the predicted yaw, pitch, and roll angles of the best image.
      * Save the best image with a drawn pose axis to `most_frontal_image.png`.
      * Apply the inverse rotation to the `your_model_normalized.obj` and save the final corrected model as `rotated_head.obj`.

### Step 4: (Optional) Generate a Realistic Dataset

The last part of the notebook provides a separate script to render a dataset from three different camera heights, introducing random jitter to simulate more realistic camera movements.

1.  **Update `model_filename`** to use the newly corrected model:
    ```python
    model_filename = '/content/rotated_head.obj'
    ```
2.  **Set `OUTPUT_FOLDER`** to your desired output location, for example, a Google Drive folder.
3.  **Configure `CAMERA_TARGET_OFFSET_Y`** to adjust the camera's vertical aim. A value of `0.15` or `0.2` is often good for models with shoulders, while `0.0` is suitable for head-only models.
4.  **Run the cell.** The script will generate a new set of images from the specified viewpoints, saving them to the output folder.

* The `SixDRepNet` model is used to predict the yaw, pitch, and roll angles for each image.
* A **"weighted score"** is calculated for each image. A lower score indicates a more frontal pose, with adjustable weights prioritizing a straight-on view.
* The image with the lowest score is selected as the best frontal image. The angles from this image are then used to reorient the original 3D model, saving the result as `rotated_head.obj`.
* An additional script generates images from different camera heights (chin, nose, eyes) with random jitter and tilt to create an even more realistic and varied dataset.
