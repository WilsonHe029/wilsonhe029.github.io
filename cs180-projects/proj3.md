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

## Part B.1: Harris Corner Detection

We first use the Harris corner algorithm to find all potential interest points in an image, and then use Adaptive Non-Maximal Suppression (ANMS) to select a fixed number of robust, spatially-distributed features.

### Adaptive Non-Maximal Suppression (ANMS)
We first use the provided single-scale Harris interest point detector to find the potential interest points. However, the initial Harris detection step often produces a large number of corners that are densely clustered in high-texture areas. This is computationally inefficient and can lead to poor matching. To solve this, we implemented Adaptive Non-Maximal Suppression (ANMS) as detailed in Section 3 of the paper.

The goal of ANMS is to select a fixed number of interest points, $n_{ip}$, that are well-distributed spatially. Our implementation follows these steps:
1.  For every interest point $x_i$, we calculate its minimum suppression radius $r_i$, the minimum Euclidean distance to any other interest point $x_j$ that has a "significantly stronger" response.
2.  The strength condition is defined as $f(x_i) < c_{robust}f(x_j)$. We used the paper's suggested value of $c_{robust} = 0.9$.
3.  After computing $r_i$ for all points, we sort the points by their suppression radii in descending order.
4.  Finally, we select the top $n_{ip}$ interest points with the largest radii. For this project, we used $n_{ip} = 500$.

### Results

We applied both our Harris detector and ANMS algorithm to the Haas image set. The results are visualized in the following figure.

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3b/part_b_1_harris_matches_haas2big_haas3big_comparison.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

The initial Harris corner detection (without suppression) identified 927 corners in the first image and 884 corners in the second. As seen in the "Non-ANMS" plots, these points are heavily clustered in textured regions like the trees, leaving large areas of the image (like the sky) with fewer features.

Next, we applied ANMS to this initial set to select the best $n_{ip} = 500$ points. The "ANMS" plots show that these points are far more spatially distributed across the entire scene. This provides a much better and more compact set of features for the subsequent matching steps.

The bottom row displays the matching point sets for both methods. 111 matches are found for the "Non-ANMS" points (many of which are likely incorrect), while 67 are found for the "ANMS" points. While its matches are still imperfect, ANMS provides a cleaner, smaller, and more robust set for geometric verification.

## Part B.2: Feature Descriptor Extraction

With a set of robust interest points selected, the next step is to create a distinctive descriptor for each point. This descriptor, a small vector of values, should uniquely identify the local image structure around the point for it to be reliably matched with corresponding points in other images. Following Section 4 of the paper, we extract a feature descriptor for each of the 500 interest points selected by ANMS.

For each interest point, we first extract a large, 40x40 pixel window centered on the point's coordinates. This 40x40 window is then downsampled to an 8x8 patch to gain robustness against small errors in interest point location. This process blurs the local structure, retaining only the dominant, low-frequency information. Finally, the 8x8 patch is normalized to have a mean of 0 and a standard deviation of 1. This bias/gain normalization makes the descriptor invariant to simple affine changes in brightness and contrast. The resulting 64-element (8x8) vector is the final feature descriptor for the interest point.

### Results

We applied the descriptor extraction process to the 500 ANMS points found in part B.1. The results are visualized in the figures below.

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3b/part_b_2_descriptor_windows_haas2big_anms.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

The first figure shows a random sample of the 40x40 source windows extracted from the image, one for each feature. These are the high-resolution patches before downsampling and normalization.

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3b/part_b_2_descriptor_patches_haas2big_anms.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

The second figure shows the normalized 8x8 descriptors corresponding to some of these windows. The blurriness of the patches is a direct result of the downsampling from 40x40 and provides robustness. The color bars, ranging approximately from -3 to 3, and the console output confirm that the patches have been successfully normalized to have a mean of zero and unit variance.

## Part B.3: Feature Matching

Once we have robust interest points and distinctive descriptors for each, the next step is to find correspondences, or "matches," between them. The goal is to compare every feature descriptor from the first image with every descriptor from the second image and keep only the pairs that are both highly similar and unambiguous.

We implemented the feature-space outlier rejection strategy, which is designed to filter out ambiguous matches. Instead of just accepting the single best match for a feature, we use the ratio test. For computational efficiency, our implementation operates on squared Euclidean distances. The algorithm is as follows: For each feature descriptor in Image 1, we find its nearest neighbor and second-nearest neighbor in Image 2 by calculating squared Euclidean distances. Let the **squared** distance to the 1-NN be $d_{1,sq}$ and the **squared** distance to the 2-NN be $d_{2,sq}$. We compute the ratio of these squared distances: $r = d_{1,sq} / d_{2,sq}$. This ratio $r$ is compared against a `ratio_threshold`. A match is accepted only if $r < \text{ratio\_threshold}$.

This test is effective because a distinctive (and correct) match should have a 1-NN distance that is significantly smaller than the distance to any other feature. This results in a low ratio. An ambiguous match (e.g., a repeating texture) will have a 1-NN and 2-NN with similar squared distances, resulting in a ratio close to 1.0. We tested several different `ratio_threshold` values (from 0.4 to 0.9) to observe the trade-off between the number of matches and their quality.

### Results

We ran our matching algorithm on the feature sets for the Haas images using the 500 ANMS points for each.

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3b/part_b_3_feature_matches_haas2big_haas3big_ratio_0.6_anms.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

The first figure shows the matching results with a strict threshold of 0.6. This produced 79 matches. Visually, these matches appear to be high-quality, with most corresponding points clearly landing on the same part of the building or landscape in both images. The lines are mostly parallel and consistent with the camera's slight shift.

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3b/part_b_3_feature_matches_haas2big_haas3big_ratio_0.8_anms.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

The second figure shows the results with a looser threshold of 0.8. This produced 159 matches. While this provides more data for stitching, the number of obvious false positives (incorrectly matched points) increases significantly.

This experiment confirms that a stricter threshold filters out more false positives at the cost of also removing some correct matches. A threshold between 0.6 and 0.8, as suggested by the paper's analysis, provides a good balance between the quantity and quality of matches.

## Part B.4: RANSAC for Robust Homography

The feature matching step from B.3 provides a decent set of matches. However, many of these are incorrect (outliers) due to ambiguous features or limitations of the ratio test. To reliably compute the homography between the images, we must find a model that is robust to this large percentage of outliers. This section implements the 4-point RANdom SAmple Consensus (RANSAC) algorithm to find a robust homography by estimating the homography matrix $H$ from the set of all matches.

The algorithm runs for a fixed number of iterations. In each iteration, 4 match pairs are randomly selected from the total set of matches. A candidate homography $H$ is computed from these 4 pairs and then used to transform all matched points from Image 1 into the coordinate system of Image 2. For each transformed point, we calculate the Euclidean distance between itself and its corresponding point in Image 2. If the distance is below a set threshold (5.0 pixels), the match is counted as an "inlier" for this model. After all iterations, the algorithm takes the largest set of inliers found and computes a final, more accurate homography $H$ using all points in this set. This homography is then used to warp Image 1 and stitch it with Image 2 using the same mosaicking code developed in Part A to create a fully automatic panorama.

### Results

We ran our pipeline on three pairs of images: Haas, desk, and home. The RANSAC algorithm was able to successfully find a homography for all three.

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3b/part_b_4_ransac_matches_haas2big_haas3big_anms.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3b/part_b_4_ransac_matches_desk2big_desk3big_anms.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3b/part_b_4_ransac_matches_home1big_home2big_anms.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

The above figures shows the RANSAC results for the three image pairs. For Haas, the algorithm identified 57 inliers (35.8%) and rejected 102 outliers from 159 initial matches. For desk, the algorithm identified 51 inliers (30.5%) and rejected 116 outliers from 167 initial matches. For home, the algorithm identified 34 inliers (17.0%) and rejected 166 outliers from 200 initial matches. For all image sets, even home that has a high number of initial matches yet many ambiguous ones, RANSAC successfully isolated the true inliers that define the correct homography. The inliers, shown in color, are clearly geometrically consistent.

These final robust homographies were then used to create the automatic mosaics. The follwoing figures provide a side-by-side comparison between the manual mosaics in Part A and the new automatic mosaics.

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3b/part_b_4_stitching_comparison_haas2big_haas3big_anms.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3b/part_b_4_stitching_comparison_desk2big_desk3big_anms.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj3b/part_b_4_stitching_comparison_home1big_home2big_anms.png" alt="Figure 15" style="max-width: 100%; height: auto;">
  </div>
</div>

In all three cases, the automatic stitching, based on a large set of inliers (34-57), produces a seamless mosaic similar to the one created with a few manually-selected points. The full pipeline successfully creates high-quality automatic panoramas.

[this tool]: https://cal-cs180.github.io/fa23/hw/proj3/tool.html
