---
title: Project 1
parent: CS 180
layout: default
nav_order: 4
---

# Project 1: Images of the Russian Empire: Colorizing the Prokudin-Gorskii photo collection

Due Date: Friday, September 12, 2025 at 11:59 PM

---

## Introduction

This project implements a colorization system for Prokudin-Gorskii's historical glass plate photographs. The challenge is to align these three color channels to reconstruct the original full-color photographs. The system processes 17 historical images (including 14 required ones and 3 of my choice) from the early 20th century, automatically detecting the optimal alignment offsets for each color channel and producing high-quality color reconstructions.

---

## Method

### Single-Scale Alignment

The fundamental approach to aligning two images is to test a range of possible offsets and select the one that makes the images most similar. The process begins by splitting the input image into three equal-sized grayscale images, representing the Blue, Green, and Red channels. The Blue channel is designated as the fixed reference. To align the Green and Red channels, we perform an exhaustive search over a pre-defined window of displacements in both the x and y directions. For each potential offset, the channel is shifted, and the similarity between the overlapping region of the shifted Green channel and the reference Blue channel is calculated. The metric used for this comparison is the L2 norm, defined as:

```python
L2 = sqrt(sum((I1[i, j] - I2[i, j])**2 for i, j in pixels))
```

Here, a lower L2 norm indicates a better match. The displacement vector that yields the minimum L2 norm is chosen as the best alignment.

While simple and effective for small images, this brute-force method is computationally infeasible for the large, high-resolution .tif files, as the number of calculations grows quadratically with the image dimensions. This limitation directly motivates the need for a more efficient, multi-scale approach.

### Multi-Scale Pyramid Alignment

To handle large images efficiently, I implemented a coarse-to-fine image pyramid algorithm. This strategy finds a rough alignment on small, downsampled versions of the images and progressively refines that alignment on higher-resolution versions, which avoids an expensive search over a large displacement window on the full-sized image.

1. **Data Pre-processing: Border Cropping**

    Noisy, inconsistent borders from the edges of the glass plates could be a significant source of alignment error. To mitigate this, the first step is to crop a fixed percentage (8%) from all sides of the B, G, and R channels. This ensures that the alignment algorithm only operates on meaningful image data.

2. **Pyramid Construction**

    I construct a pyramid for each color channel by repeatedly downsampling the image by a factor of two. The process stops when the smallest dimension of the image is less than 32 pixels. The downsampling is performed by a resize function that first applies a simple 3x3 box blur to the image to reduce aliasing artifacts before subsampling the pixels.

3. **Coarse-to-Fine Alignment with L2 Norm**

    The alignment process starts at the coarsest level of the pyramid (the smallest images) and iterates down to the finest (full-resolution) level.

    - Step 1: At the coarsest level, an exhaustive search is performed over a wide window of [-15, 15] pixels to find the best initial offset between the channels. Because the images are very small, this search is computationally cheap.

    - Step 2: The displacement vector found at this level is then doubled and passed down to the next, finer level to serve as an initial estimate.

    - Step 3: At the finer level, a new, much smaller search is performed locally around the propagated estimate. This refines the alignment with greater precision.

    - Step 4: Steps 2 and 3 are repeated until the final, full-resolution level is reached, yielding a highly accurate displacement vector.

4. **Feature Selection with a Sobel Filter**

    Aligning the raw pixel intensities using L2 norm or NCC failed on certain images, most notably the Emir.

    | Colorized Result | Green Offset | Red Offset |
    |------------------|-------------|------------|
    | ![Emir]({{ '/assets/images/proj1/output_emir.jpeg' | relative_url }})<br/><center>[Required, Naive] emir.tif</center> | (49, 24) | (393, -573) |

    The Emir's brightly colored clothing creates vastly different intensity patterns in the red, green, and blue channels, confusing any metric based on raw brightness. However, the underlying structural edges—the outlines of his clothes, his beard, and the patterns on the wall—are much more consistent across the channels.

    To exploit this, I switched from aligning raw intensities to aligning edge maps. Before building the pyramid, I first process each color channel with a Sobel filter. The Sobel operator approximates the gradient of the image's intensity, producing a new image where the brightness of each pixel corresponds to the strength of an edge at that location. The alignment pyramid is then built from these edge maps. This forces the algorithm to match the consistent structural features in the scene, making it robust to the large color and brightness variations.

---

## Results

The system successfully processed all 17 test images with the following results, all of which achieving good alignment.

| Colorized Result | Green Offset | Red Offset |
|------------------|-------------|------------|
| ![Cathedral]({{ '/assets/images/proj1/output_cathedral.jpeg' | relative_url }})<br/><center>[Required] cathedral.jpg</center> | (5, 2) | (12, 3) |
| ![Church]({{ '/assets/images/proj1/output_church.jpeg' | relative_url }})<br/><center>[Required] church.tif</center> | (25, 4) | (58, -4) |
| ![Emir]({{ '/assets/images/proj1/output_emir_sobel.jpeg' | relative_url }})<br/><center>[Required, Sobel] emir.tif</center> | (49, 23) | (107, 40) |
| ![Harvesters]({{ '/assets/images/proj1/output_harvesters.jpeg' | relative_url }})<br/><center>[Required] harvesters.tif</center> | (59, 17) | (123, 13) |
| ![Icon]({{ '/assets/images/proj1/output_icon.jpeg' | relative_url }})<br/><center>[Required] icon.tif</center> | (41, 17) | (89, 23) |
| ![Italil]({{ '/assets/images/proj1/output_italil.jpeg' | relative_url }})<br/><center>[Required] italil.tif</center> | (38, 21) | (76, 35) |
| ![Lastochikino]({{ '/assets/images/proj1/output_lastochikino.jpeg' | relative_url }})<br/><center>[Required] lastochikino.tif</center> | (-3, -2) | (75, -9) |
| ![Lugano]({{ '/assets/images/proj1/output_lugano.jpeg' | relative_url }})<br/><center>[Required] lugano.tif</center> | (41, -16) | (93, -29) |
| ![Melons]({{ '/assets/images/proj1/output_melons.jpeg' | relative_url }})<br/><center>[Required] melons.tif</center> | (83, 11) | (179, 13) |
| ![Monastery]({{ '/assets/images/proj1/output_monastery.jpeg' | relative_url }})<br/><center>[Required] monastery.jpg</center> | (-3, 2) | (3, 2) |
| ![Self Portrait]({{ '/assets/images/proj1/output_self_portrait.jpeg' | relative_url }})<br/><center>[Required] self_portrait.tif</center> | (78, 29) | (176, 37) |
| ![Siren]({{ '/assets/images/proj1/output_siren.jpeg' | relative_url }})<br/><center>[Required] siren.tif</center> | (49, -6) | (95, -25) |
| ![Three Generations]({{ '/assets/images/proj1/output_three_generations.jpeg' | relative_url }})<br/><center>[Required] three_generations.tif</center> | (53, 13) | (112, 11) |
| ![Tobolsk]({{ '/assets/images/proj1/output_tobolsk.jpeg' | relative_url }})<br/><center>[Required] tobolsk.jpg</center> | (3, 3) | (6, 3) |
| ![Master Prok 00300]({{ '/assets/images/proj1/output_master-pnp-prok-00300-00398u.jpeg' | relative_url }})<br/><center>[My choice] Station</center> | (54, 25) | (113, 36) |
| ![Master Prok 00400]({{ '/assets/images/proj1/output_master-pnp-prok-00400-00458u.jpeg' | relative_url }})<br/><center>[My choice] Locomotive</center> | (42, 5) | (87, 32) |
| ![Master Prok 00700]({{ '/assets/images/proj1/output_master-pnp-prok-00700-00704u.jpeg' | relative_url }})<br/><center>[My choice] Big Cat</center> | (59, 21) | (130, 27) |
