---
title: Project 3
parent: CS 180
layout: default
nav_order: 6
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

# Project 3: Stitching Photo Mosaics

Due Date [Part 1]: Wednesday, October 8, 2025 at 11:59pm

Due Date [Part 2]: Friday, October 17, 2025 at 11:59pm

---

## Part A.1: Shoot the Pictures

I took three sets of images for creating photo mosaics. The first set was taken outdoors on my way to lecture of the Haas School of Business, the second at my home in the kitchen and dining room, and the third of my working space. The photographs were taken by rotating the camera around a center of projection, keeping the camera's position fixed. There is significant overlap between consecutive photos to successful stitching.

**Image Set 1: Haas**

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj3/haas1.jpeg" alt="Haas 1" style="width: 32%; height: auto;">
  <img src="/assets/images/proj3/haas2.jpeg" alt="Haas 2" style="width: 32%; height: auto;">
  <img src="/assets/images/proj3/haas3.jpeg" alt="Haas 3" style="width: 32%; height: auto;">
</div>

**Image Set 2: Home**

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj3/home1.jpeg" alt="Home 1" style="width: 32%; height: auto;">
  <img src="/assets/images/proj3/home2.jpeg" alt="Home 2" style="width: 32%; height: auto;">
  <img src="/assets/images/proj3/home3.jpeg" alt="Home 3" style="width: 32%; height: auto;">
</div>

**Image Set 3: Desk**

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj3/desk1.jpeg" alt="Home 1" style="width: 32%; height: auto;">
  <img src="/assets/images/proj3/desk2.jpeg" alt="Home 2" style="width: 32%; height: auto;">
  <img src="/assets/images/proj3/desk3.jpeg" alt="Home 3" style="width: 32%; height: auto;">
</div>

## Part A.2: Recover Homographies

To align and stitch the images, we try to recover the transformation between each image pair. This transformation is a homography, a 3x3 matrix $H$ with 8 degrees of freedom, which maps points from one image plane to another according to the equation $p' = Hp$. The homography is computed from a set of corresponding points between the two images.

### Point Correspondence Visualizations

The point correspondences were selected manually across image pairs with [this tool] made my a previous student. These points serve as the input for the `computeH` function.

**Haas Image Set Correspondences**

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3/correspondences_haas_set.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

**Home Image Set Correspondences**

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3/correspondences_home_set.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

**Desk Image Set Correspondences**

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3/correspondences_desk_set.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

### System of Equations

For each point correspondence $(x, y) \to (x', y')$, we derive two linear equations. The relationship is given by:

$$x' = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + 1} \quad \text{and} \quad y' = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + 1}$$

These are rearranged into a linear form suitable for least-squares, $Ah = b$:

$$
\begin{bmatrix}
x_1 & y_1 & 1 & 0 & 0 & 0 & -x_1x'_1 & -y_1x'_1 \\
0 & 0 & 0 & x_1 & y_1 & 1 & -x_1y'_1 & -y_1y'_1 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
x_n & y_n & 1 & 0 & 0 & 0 & -x_nx'_n & -y_nx'_n \\
0 & 0 & 0 & x_n & y_n & 1 & -x_ny'_n & -y_ny'_n
\end{bmatrix}
\begin{bmatrix}
h_{11} \\ h_{12} \\ h_{13} \\ h_{21} \\ h_{22} \\ h_{23} \\ h_{31} \\ h_{32}
\end{bmatrix}
=
\begin{bmatrix}
x'_1 \\ y'_1 \\ \vdots \\ x'_n \\ y'_n
\end{bmatrix}
$$

For example, between `haas1.jpeg` and `haas2.jpeg`, there are seven corresponding points, which are:

- $(132, 225) \to (4, 226)$
- $(219, 267) \to (89, 264)$
- $(207, 178) \to (90, 178)$
- $(171, 131) \to (62, 129)$
- $(127, 34) \to (20, 14)$
- $(215, 112) \to (105, 118)$
- $(227, 192) \to (107, 193)$

In this case, the matrix $A$ and vector $b$ would be:

$$
A =
\begin{bmatrix}
132 & 225 & 1 & 0 & 0 & 0 & -528 & -900 \\
0 & 0 & 0 & 132 & 225 & 1 & -29832 & -50850 \\
219 & 267 & 1 & 0 & 0 & 0 & -19491 & -23763 \\
0 & 0 & 0 & 219 & 267 & 1 & -57816 & -70488 \\
207 & 178 & 1 & 0 & 0 & 0 & -18630 & -16020 \\
0 & 0 & 0 & 207 & 178 & 1 & -36846 & -31684 \\
171 & 131 & 1 & 0 & 0 & 0 & -10602 & -8122 \\
0 & 0 & 0 & 171 & 131 & 1 & -22059 & -16899 \\
127 & 34 & 1 & 0 & 0 & 0 & -2540 & -680 \\
0 & 0 & 0 & 127 & 34 & 1 & -1778 & -476 \\
215 & 112 & 1 & 0 & 0 & 0 & -22575 & -11760 \\
0 & 0 & 0 & 215 & 112 & 1 & -25370 & -13216 \\
227 & 192 & 1 & 0 & 0 & 0 & -24289 & -20544 \\
0 & 0 & 0 & 227 & 192 & 1 & -43811 & -37056
\end{bmatrix}, b =
\begin{bmatrix}
4 \\ 226 \\ 89 \\ 264 \\ 90 \\ 178 \\ 62 \\ 129 \\ 20 \\ 14 \\ 105 \\ 118 \\ 107 \\ 193
\end{bmatrix}
$$

### Solving for `H`

The homography matrix $H$ is recovered using a function `computeH(im1_pts, im2_pts)`. This function takes two $n \times 2$ arrays of corresponding $(x, y)$ coordinates and returns the computed $3 \times 3$ homography matrix. The function sets up a system of linear equations as outlined above and solves for the 8 unknown parameters of $H$ using the least-squares method.

### Recovered Homography Matrices

The `computeH` function was run on the selected correspondences for each image pair, producing the following homography matrices.

* **Pair: haas1_haas2**

  $$
  H = \begin{bmatrix}
  1.6448 & -0.1674 & -174.6953 \\
  0.5070 & 1.4771 & -96.3492 \\
  0.0024 & 0.0001 & 1.0
  \end{bmatrix}
  $$

* **Pair: haas2_haas3**

  $$
  H = \begin{bmatrix}
  1.7645 & -0.1201 & -204.0111 \\
  0.6346 & 1.5495 & -113.5978 \\
  0.0032 & -0.0001 & 1.0
  \end{bmatrix}
  $$

* **Pair: home1_home2**

  $$
  H = \begin{bmatrix}
  1.6135 & 0.0510 & -219.1465 \\
  0.3225 & 1.3772 & -43.6618 \\
  0.0024 & -0.0002 & 1.0
  \end{bmatrix}
  $$

* **Pair: home2_home3**

  $$
  H = \begin{bmatrix}
  1.5028 & 0.0668 & -198.2535 \\
  0.2896 & 1.3806 & -47.8110 \\
  0.0022 & 0.0000 & 1.0
  \end{bmatrix}
  $$

* **Pair: desk1_desk2**

  $$
  H = \begin{bmatrix}
  1.1162 & 0.0081 & -144.3589 \\
  0.0702 & 1.0794 & -10.5469 \\
  0.0005 & -0.0000 & 1.0
  \end{bmatrix}
  $$

* **Pair: desk2_desk3**

  $$
  H = \begin{bmatrix}
  1.0526 & 0.0080 & -122.0763 \\
  0.0353 & 1.0048 & -4.7699 \\
  0.0003 & -0.0002 & 1.0
  \end{bmatrix}
  $$

## Part A.3: Warp the Images

### Warping Methodology

Once a homography matrix is known, it can be used to warp an image. For each pixel in the target (output) image, the inverse homography ($H^{-1}$) is used to calculate its corresponding coordinate in the source (input) image. Since this coordinate is a floating-point value and does not land perfectly on a single pixel, an interpolation method is required to determine its color. We implement two distinct interpolation methods.

* **Nearest Neighbor Interpolation:** This is the simplest method. The algorithm calculates the source coordinate and rounds it to the nearest integer coordinates. It then samples the color from that single, closest pixel.
* **Bilinear Interpolation:** The algorithm finds the four source pixels surrounding the calculated coordinate. It then performs a weighted average of the colors of these four pixels, with the weights determined by the sub-pixel distance to each neighbor.

### Image Rectification Examples

Image rectification is a practical application of warping. By selecting four corners of a rectangular object in a photo and defining a target set of four points as a perfect rectangle, a homography can be computed to "straighten" the object.

**Example 1: Box**

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3/rectification_box.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

**Example 2: Door**

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3/rectification_door.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

### Interpolation Method Comparison

Nearest Neighbor is much faster because it only requires one pixel lookup per output pixel. However, this speed comes at aliasing. Bilinear Interpolation is slower because it performs four pixel lookups and a weighted average for each output pixel, but it results in a far superior image with smooth edges and fewer artifacts.

**Visual Quality:** Bilinear interpolation consistently produces a higher quality, smoother image. We zoom into the white cheddar crunchers can in the box example to visualize this difference.

<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="/assets/images/proj3/rectification_box_bilinear_detail.png" alt="Haas 1" style="width: 48%; height: auto;">
  <img src="/assets/images/proj3/rectification_box_nn_detail.png" alt="Haas 2" style="width: 48%; height: auto;">
</div>
<div style="text-align: center; margin-top: 8px;">
  <em>Warping Details: Bilinear (left) vs Nearest Neighbor (right)</em>
</div>

Here, the small text beneath the word "White Cheddar Crunchers" remain reasonably smooth and recognizable for Biliear, yet they are jagged and blocky to the degree that they are nearly ilegible for Nearest Neighbor.

**Speed:** Bilinear Interpolation is noticeable slower than Nearest Neighbor, since it samples from four pixels instead of one. For the box, Nearest Neighbor took 2.13s, whereas Bilinear Interpolation took 13.24s. For the door, Nearest Neighbor took 0.12s, while Bilinear Interpolation took 0.67s.

## Part A.4: Blend the Images into a Mosaic

The final step is to combine the warped images into a single, seamless panoramic mosaic. The mosaics were created with the following procedures.

1. For each image set, the central image is chosen as the reference plane. The reference image itself is not warped; its transformation is the identity matrix. All other images in the sequence are transformed to align with this reference image's perspective.

2. The pairwise homographies calculated in Part A.2 (which map image `i` to `i+1`) are chained together to create cumulative homographies. For images to the left of the reference, the homographies are composed sequentially. For images to the right, the inverse of the homographies are composed.

3. Each image is warped using its corresponding cumulative homography. To maintain high visual quality and avoid aliasing, the bilinear interpolation method from Part A.3 is used for this process.

4. To eliminate hard edges in the final mosaic, a blending strategy based on weighted averaging is employed. A soft mask is generated for each warped image where pixels near the center of the image have a weight close to 1, and the weights fall off linearly towards 0 at the edges.

5. A final canvas is created that is large enough to contain all the warped images. The final color of each pixel on the canvas is determined by calculating the weighted average of all contributing warped images at that location, using their alpha masks as the weights.

### Mosaic Results

**Mosaic 1: Haas**

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3/mosaic_haas_result.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

**Mosaic 2: Home**

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3/mosaic_home_result.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

**Mosaic 3: Desk**

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3/mosaic_desk_result.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

[this tool]: https://cal-cs180.github.io/fa23/hw/proj3/tool.html
