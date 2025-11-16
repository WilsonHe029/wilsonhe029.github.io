---
title: Project 4
parent: CS 180
layout: default
nav_order: 7
---

<script>
  window.MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']]
    },
    options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }
  };
  </script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# Project 4: Neural Radiance Field

Due Date: Friday, November 14, 2025 at 11:59pm

# Part 0: Calibrating Your Camera and Capturing a 3D Scan

This initial part of the assignment involves setting up the necessary components for building a Neural Radiance Field (NeRF). This process is divided into two main steps: first, calibrating our camera to determine its intrinsic parameters, and second, capturing a 3D scan of an object using ArUco tags for pose estimation.

---

## Part 0.1 & 0.2: Calibrating Your Camera & Capturing a 3D Object Scan

The primary goal of this step is to find the camera's intrinsic parameters, specifically the camera matrix (also known as intrinsics) and the distortion coefficients. These parameters model how the 3D world is projected onto the 2D image plane and are essential for accurately mapping 3D points to 2D pixels and vice versa.

### Images

We began by capturing a set of 40 images of the provided ArUco calibration tag sheet using a phone camera at various angles, distances, and orientations while maintaining a constant zoom level. The next goal was to capture a dataset of images of a specific object (a Ferrari F1 model car) from which we will generate a NeRF. We placed the target object on a tabletop next to a printed ArUaCo tag. Then, using the same camera and zoom level from the calibration step, we captured 45 images of the object, moving the camera around the object to capture various angles.

Below are one example of the calibration image set and the object image set each.

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj4/IMG_7330.jpeg" alt="Calibration image example" style="width: 48%; height: auto;">
  <img src="/assets/images/proj4/IMG_7402.jpeg" alt="Object image example" style="width: 48%; height: auto;">
</div>


### Calibration

For all calibration images, we detect the ArUco markers using `cv2.aruco.detectMarkers`. For each detected tag, we defined its 3D world coordinates on the $Z=0$ plane, with a specified physical size of 0.02m. The script accumulates these 3D object points and their corresponding 2D image pixel coordinates. This function computed the camera's parameters by minimizing the reprojection error. The returned results include:
* **Camera Matrix ($K$)**: A 3x3 matrix containing the focal lengths ($f_x$, $f_y$) and the principal point ($c_x$, $c_y$).
    $$
    K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
    $$
* **Distortion Coefficients ($D$)**: A vector of coefficients $(k_1, k_2, p_1, p_2, ...)$ that model the lens distortion.
* **Mean Reprojection Error**: A scalar value indicating the average pixel distance between the projected 3D points and their detected 2D counterparts.

These results were saved to a file named `results/camera_calibration.npz` for use in the next steps.

## Part 0.3: Estimating Camera Pose

The goal of this part is to estimate the precise camera extrinsics (position and orientation) for each image of our object scan given their known 3D world points and their 2D projections in an image.

### Pose Estimation

To find the camera pose for each image, we first loaded the `camera_matrix` ($K$) and `dist_coeffs` ($D$) obtained from Part 0.1. For each image, we performed the following steps:

1.  Find the 2D pixel coordinates (`imagePoints`) of the single ArUco tag's corners.
2.  Define the 3D coordinates (`objectPoints`) of the tag's corners in the world space assuming it lies on the $Z=0$ plane.
3.  Cpmpute a rotation vector (`rvec`) and a translation vector (`tvec`) with the `cv2.solvePnP` function that define the world-to-camera transformation.
4.  The output from `cv2.solvePnP` describes how to transform points from the world (tag) coordinate system to the camera's coordinate system. We obtained the camera-to-world (`c2w`) matrix by first converting `rvec` to a 3x3 rotation matrix `R` using `cv2.Rodrigues`, forming the 4x4 world-to-camera matrix, and then computing its inverse.

This process was repeated for all images where the tag was successfully detected, and the resulting `c2w` matrices were collected and saved to `results/camera_poses.npz`.

### Visualization

To verify our pose estimation, we used the `viser` library to visualize the camera frustums in 3D. We loaded each image and its corresponding `c2w` matrix to generate a 3D point cloud of all camera poses. The screenshots below show the resulting camera frustums.

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj4/part03-1.png" alt="Camera frustum visualization 1" style="width: 48%; height: auto;">
  <img src="/assets/images/proj4/part03-2.png" alt="Camera frustum visualization 2" style="width: 48%; height: auto;">
</div>

## Part 0.4: Undistorting images and creating a dataset

The final step was to undistort the captured images using the calibration parameters and package them, along with their corresponding poses, into a single `.npz` dataset file suitable for NeRF training. We follow the algorithm described below:

1.  Load the `camera_matrix` and `dist_coeffs` from the `results/camera_calibration.npz` file generated in Part 0.1. The focal length is extracted from the camera matrix (assuming $f_x \approx f_y$).
2.  For each image in the `object/` directory, perform two operations:
    * Calculate the camera's `rvec` and `tvec`. This result is converted into a 4x4 camera-to-world (`c2w`) transformation matrix.
    * Remove any lens distortion with `cv2.undistort`.
3.  Convert the undistorted images, which are in OpenCV's default BGR format, to RGB.
4.  The matched set of undistorted RGB images and their `c2w` poses is randomly shuffled. It is then split into training (80%), validation (10%), and test (10%) sets.
5.  Finally, save these data into a single `results/my_data.npz`. This file contains the following keys:
* `images_train`: A NumPy array of shape (N\_train, H, W, 3)
* `c2ws_train`: A NumPy array of shape (N\_train, 4, 4)
* `images_val`: A NumPy array of shape (N\_val, H, W, 3)
* `c2ws_val`: A NumPy array of shape (N\_val, 4, 4)
* `c2ws_test`: A NumPy array of shape (N\_test, 4, 4)
* `focal`: A float value for the camera's focal length.

## Part 1: Fit a Neural Field to a 2D Image

The goal is to train a network that learns a continuous mapping from 2D pixel coordinates $(x, y)$ to their corresponding 3D color values $(R, G, B)$.s

### Architecture

We defined a `NeuralField2D` model, which is an 8-layer MLP with a hidden dimension of 256. Each hidden layer uses a ReLU activation function, and the final output layer uses a Sigmoid function to constrain the RGB color values to the range $[0, 1]$.

To enable the network to learn high-frequency details (like the fox's fur), we do not feed the 2D coordinates directly to the MLP. Instead, we first pass them through a `PositionalEncoding` layer. This layer maps the 2D input coordinates $(x, y)$ to a higher-dimensional feature vector. We use $L=10$ frequency levels. This encoding includes the original 2 coordinates plus 4 components $(\sin, \cos$ for $x$ and $y$) for each of the 10 frequencies, resulting in an input vector of $2 + 4 \times 10 = 42$ dimensions for the MLP.

We implemented an `ImageDataset` that loads the target image and provides a `__getitem__` method to randomly sample a `batch_size` of 10,000 pixels (coordinate-color pairs) for each training iteration. The network is trained for 2,000 iterations using the Adam optimizer with a learning rate of $1 \times 10^{-2}$.

We use the Mean Squared Error (MSE) between the network's predicted pixel colors and the ground-truth colors as our loss function. We also track the Peak Signal-to-Noise Ratio (PSNR) as our primary quality metric, calculated as $PSNR = -10 \log_{10}(MSE)$.

### Results

We trained our model on the provided image of a fox. As seen in the progression image, the network starts by outputting a mean-color (gray) image at iteration 0. It quickly learns the low-frequency components, producing a blurry version of the fox by iteration 100. As training progresses, it resolves higher-frequency details, with the fur and background becoming significantly sharper by iterations 1000 and 2000.

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj4/part11_example_progression.png" alt="Training progression from iteration 0 to 2000" style="width: 100%; height: auto;">
</div>

Our final trained model achieves a PSNR of 24.93 dB. The PSNR curve shows a rapid increase in quality for the first ~250 iterations, followed by a much slower, steadier improvements as the model refines the high-frequency details. The final rendered image is a high-fidelity reconstruction of the original.

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj4/part11_example_result.png" alt="Side-by-side of original and final rendered image" style="width: 100%; height: auto;">
</div>

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj4/part11_psnr_curve.png" alt="PSNR training curve" style="width: 80%; height: auto;">
</div>

### Hyperparameter Tuning

We also investigated the impact of the MLP width (`hidden_dim`) and the number of positional encoding frequencies (`L`). We found out that `Width=128, L=10` gave an optimal PSNR of **25.98 dB**, and the image is sharp and well-reconstructed.

<div style="display: flex; justify-content: center; gap: 10px;">
    <img src="/assets/images/proj4/part11_hyperparameter_grid.png" alt="2x2 grid comparing network width and L value" style="width: 100%; height: auto;">
</div>

This experiment shows that the number of PE frequencies ($L$) is critical for capturing high-frequency detail ($L=10$ is clearly better than $L=5$). It also suggests that a wider network is not necessarily better; the `Width=256, L=10` model may have been more difficult to train or more prone to fitting noise, leading to a lower-quality result than its `Width=128` counterpart.

## Part 2.1: Create Rays from Cameras

To fit a Neural Radiance Field, we must convert 2D pixel coordinates into 3D rays (origins and directions) in the world coordinate system. This part involved implementing the core geometric functions to generate these rays from our camera parameters.

We implemented a fully vectorized `pixel_to_ray` function which, for a given pixel `uv`, computes its corresponding 3D ray in world coordinates. The ray origin ($\mathbf{r_o}$) is simply the camera's 3D position, while the ray direction ($\mathbf{r_d}$) is found with the following process. We first find a 3D point on the ray at a unit depth ($s=1$) in the camera's coordinate system using the inverse of the intrinsic matrix $\mathbf{K}$. This point is then transformed into world coordinates using the `c2w` matrix. The final direction $\mathbf{r_d}$ is the normalized vector pointing from the origin $\mathbf{r_o}$ to this transformed world point.

## Part 2.2: Sampling

The next step is to create a data pipeline that feeds our NeRF model. This involves two stages of sampling: sampling rays from our set of training images, and sampling 3D points along each of those rays.

To sample rays from images, we first generate a complete grid of all $N \times H \times W$ potential rays from all training images. During training, we randomly sample `n_rays` from this set. For each sampled index, we retrieve its 2D coordinate `uv`, ground-truth `color`, and camera pose `c2w`. These batches are then passed to our `pixel_to_ray` function.

To sample points along rays, we discretize each continuous ray by generating a number of depth values $t$ linearly spaced between the `near` and `far` scene bounds. To prevent overfitting to fixed locations, we use stratified sampling by adding a small random perturbation to each $t$ value during training. The final 3D points are then calculated using the ray equation $\mathbf{x}(t) = \mathbf{r_o} + t \mathbf{r_d}$.

## Part 2.3: Putting the Dataloading All Together

We combine all previous components into a single dataloader class `RaysData` that precomputes all data for efficient training. At initialization, the class iterates through all $N \times H \times W$ pixels in the dataset. It generates a "float grid" (with 0.5 offset) for ray generation and an "integer grid" for color lookup. Using our `pixel_to_ray` function, it computes and stores every ray's origin, direction, and corresponding pixel color into three large NumPy arrays: `self.rays_o`, `self.rays_d`, and `self.pixels`. The `sample_rays` method simply draws `n_rays` random indices from these precomputed arrays,.

## Part 2.4: Neural Radiance Field

In this part, we designed a Multi-Layer Perceptron (MLP) to learn a continuous 5D function that maps a 3D coordinate $\mathbf{x} = (x, y, z)$ and a 2D viewing direction $\mathbf{d}$ to a color $\mathbf{c} = (R, G, B)$ and a volume density $\sigma$.

### Architecture

The 3D coordinates $\mathbf{x}$ are passed through a positional encoding with $L=10$ frequencies, mapping them to a high-dimensional feature vector $\gamma(\mathbf{x})$. The 3D directions $\mathbf{d}$ are passed through a separate encoding with $L=4$ frequencies, mapping them to $\gamma(\mathbf{d})$.

The core is an 8-layer MLP (256 wide, ReLU activations). The encoded coordinates $\gamma(\mathbf{x})$ are passed through the first 4 layers. The output is then concatenated with the original $\gamma(\mathbf{x})$ (a skip connection) and passed through the final 4 layers.

The network produces two separate outputs:
    1.  **Density ($\sigma$)**: The main MLP's feature vector is passed through a linear layer and a **ReLU** activation to produce a single, non-negative, view-independent density value.
    2.  **Color ($\mathbf{c}$)**: The feature vector is first concatenated with the encoded viewing direction $\gamma(\mathbf{d})$. This combined vector is passed through another small MLP which outputs a 3-vector. A Sigmoid activation is applied to constrain the RGB color values to the range $[0, 1]$.

## Part 2.5: Volume Rendering and Training

We proceeded to train the Neural Radiance Field. This section covers both the rendering equation that enables training and the training results on the Lego and our custom F1 car datasets.

### Volume Rendering

We implemented the discrete volume rendering equation, $C(\mathbf{r}) = \sum_{i=1}^{N} T_i (1 - \exp(-\sigma_i \delta_i)) \mathbf{c}_i$, as a differentiable `volume_render` function. This function takes the MLP's output (`sigmas` and `rgbs`) and calculates the final color by:

1.  Computing the opacity $\alpha_i = 1 - \exp(-\sigma_i \delta_i)$ for each sample.
2.  Computing the transmittance $T_i = \exp(-\sum_{j=1}^{i-1} \sigma_j \delta_j)$ for each sample using `torch.cumsum`.
3.  Calculating the final sample weight $w_i = T_i \alpha_i$.
4.  Summing the weighted colors $C(\mathbf{r}) = \sum w_i \mathbf{c}_i$ to get the final pixel color.

### Training Loop

For each training iteration, we perform the following steps:
1.  **Sample Rays**: The `RaysData` loader (Part 2.3) samples a batch of 4096 rays.
2.  **Sample Points**: `sample_points_along_rays` (Part 2.2) generates 64 perturbed points along each ray.
3.  **Network Query**: The `NeRF` model (Part 2.4) queries all points and directions, outputting densities and colors.
4.  **Render & Loss**: The `volume_render` function computes the final colors, and the MSE loss is calculated against the ground-truth pixels.
5.  **Optimize**: The loss is backpropagated, and an Adam optimizer (learning rate $5 \times 10^{-4}$) updates the model weights.

### Results on Lego Dataset (Baseline)

We used the suggested scene bounds of `near=2.0` and `far=6.0`, with 64 samples per ray. to train on the `lego` dataset, which was successful and serves as a strong baseline. The training loss steadily decreased, and the validation PSNR (Peak Signal-to-Noise Ratio) consistently increased, peaking above 26 dB. The intermediate renders show the model progressing from a translucent, blurry cloud at 500 iterations to a sharp, well-defined object at 9,500 iterations.

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj4/part26_training_curves_lego.png" alt="Lego training curves" style="width: 100%; height: auto;">
</div>
<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj4/part26_iter_00500_lego.png" alt="Lego render at 500 iterations" style="width: 48%; height: auto;">
  <img src="/assets/images/proj4/part26_iter_03500_lego.png" alt="Lego render at 9500 iterations" style="width: 48%; height: auto;">
</div>
<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj4/part26_iter_06500_lego.png" alt="Lego render at 9500 iterations" style="width: 48%; height: auto;">
  <img src="/assets/images/proj4/part26_iter_09500_lego.png" alt="Lego render at 9500 iterations" style="width: 48%; height: auto;">
</div>

### Results on Custom F1 Car Dataset

To train on our custom F1 car dataset, We changed the sampling range from `[2.0, 6.0]` to `near=0.02` and `far=0.5` and increased the samples per ray to 64 to try and capture more detail to account for the closer distance between the camera and the object. Training on this dataset proved to be much more challenging, and the results failed to converge to a coherent 3D representation.

The training loss curve decreased, but the validation PSNR peaked very early (around iteration 1,000 at a low ~16 dB) and then steadily decreased for the rest of the training. This is a clear sign that the model learned a very coarse "blob" and then began to diverge. The 500-iteration render is a blurry red blob, which corresponds to the model's "best" attempt. The 9,500-iteration render is a noisy, cloudy mess, which is visually worse and reflects the collapsed PSNR.

We attribute this failure to several key challenges with our real-world data:
1.  **Reflections**: The F1 car has shiny surfaces. NeRF struggles with reflections as they violate the assumption of a static, view-independent color field. The model tried to represent these reflections as "glowing clouds" in 3D space.
3.  **Background**: The wooden table is a high-frequency, reflective background, which adds significant complexity compared to the simple black background of the Lego scene.
2.  **Pose Inaccuracy**: NeRF is extremely sensitive to pose accuracy. Any small jitter or error from our ArUco tag system (Part 0.3) can prevent the model from converging on a sharp representation.

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj4/part26_training_curves_f1.png" alt="Lego training curves" style="width: 100%; height: auto;">
</div>
<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj4/part26_iter_00500_f1.png" alt="Lego render at 500 iterations" style="width: 48%; height: auto;">
  <img src="/assets/images/proj4/part26_iter_03500_f1.png" alt="Lego render at 9500 iterations" style="width: 48%; height: auto;">
</div>
<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj4/part26_iter_06500_f1.png" alt="Lego render at 9500 iterations" style="width: 48%; height: auto;">
  <img src="/assets/images/proj4/part26_iter_09500_f1.png" alt="Lego render at 9500 iterations" style="width: 48%; height: auto;">
</div>

### Novel View Generation

We implemented a `create_novel_view_gif` function that generates new camera-to-world matrices for a circular "orbit" around the object, all looking at the origin. We then render a frame for each pose.

* **Lego GIF**: we were unable to get the GIF rotating on the correct axle, and it pans around the rear of the bulldozer instead of the top. Nevertheless, as expected from the successful training, the GIF shows a stable, coherent 3D object.
* **F1 Car GIF**: The F1 car GIF shows a rotating, "cloudy" mass of colors that does not resemble the object, which reflects the failed training.

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj4/part26_gif_lego.gif" alt="Lego novel view gif" style="width: 48%; height: auto;">
  <img src="/assets/images/proj4/part26_gif_f1.gif" alt="F1 car novel view gif" style="width: 48%; height: auto;">
</div>

