---
title: Project 2
parent: CS 180
layout: default
nav_order: 5
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

# Project 2: Fun with Filters and Frequencies!


Due Date: Friday, Sep 26, 2025 at 11:59pm

---

## Part 1.1: Convolutions from Scratch

We implement the convolution operation in two ways: a naive four-loop version and a partially-vectorized two-loop version. We then compare their output and performance against the highly optimized scipy.signal.convolve2d function and apply them to an image using a box filter for blurring and finite difference operators for edge detection.

### 2D Convolution Implementation

- **Naive Four-Loop Implementation**

    This is the most direct interpretation of convolution, which uses four nested for loops: two to iterate over the output pixel coordinates `(i, j)` and two to iterate over the kernel coordinates `(k, l)`. At the innermost loop, it performs a single multiplication and addition.

    ```python
    def convolution_four_loops(image, kernel, padding=0):
        # 4-loop convolution using explicit element-wise multiplication
        if padding > 0:
            padded_image = np.pad(image, padding, mode='constant', constant_values=0)
        else:
            padded_image = image.copy()
        img_h, img_w = padded_image.shape
        kernel_h, kernel_w = kernel.shape
        out_h = img_h - kernel_h + 1
        out_w = img_w - kernel_w + 1
        output = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                for k in range(kernel_h):
                    for l in range(kernel_w):
                        output[i, j] += padded_image[i + k, j + l] * kernel[k, l]
        return output
    ```

- **Partially-Vectorized Two-Loop Implementation**

    This version takes advantage of NumPy's vectorized operations to improve upon the naive approach. The two outer loops still iterate over the output pixel coordinates, but the inner two loops are replaced by extracting an entire image patch and computing the dot product with the kernel in one `np.sum(patch * kernel)` operation.

    ```python
    def convolution_two_loops(image, kernel, padding=0):
        # 2-loop convolution using vectorized patch dot product per location
        if padding > 0:
            padded_image = np.pad(image, padding, mode='constant', constant_values=0)
        else:
            padded_image = image.copy()
        # ... (dimensions setup) ...
        output = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                # Extract patch from image at current position
                patch = padded_image[i:i+kernel_h, j:j+kernel_w]
                # Compute dot product between patch and kernel
                output[i, j] = np.sum(patch * kernel)
        return output
    ```

### Filter Definitions

We created three distinct filters to apply to our input image:

- **Box Filter**: A 9x9 box filter, which is a normalized matrix of ones, is created to perform blurring. When convolved with an image, it replaces each pixel with the average of its 9x9 neighborhood, resulting in a smoothing effect.

- **Finite Difference Operators**: To detect edges, we used finite difference operators, which approximate the image gradient in the horizontal (D_x) and vertical (D_y) directions.

    - $D_x = [1, 0, -1]$: This kernel highlights vertical edges by computing the difference between the pixels to the left and right of the center.

    - $D_y = [1, 0, -1]^\top$: This kernel highlights horizontal edges by computing the difference between the pixels above and below the center.
    
### Comparison with `scipy.signal.convolve2d`

To validate our implementations, we compared their output with SciPy's `convolve2d` function using the `mode='valid'` setting, which corresponds to convolution without padding.

- **Correctness**: Both our four-loop and two-loop implementations produced results that are visually similar to SciPy's. It is worth noting that `scipy.signal.convolve2d performs` cross-correlation, which is equivalent to convolution with a flipped kernel. Our implementation also computes cross-correlation. For symmetric kernels like the box filter, this is irrelevant. For asymmetric kernels like the finite difference operators, our results matched SciPy's after a sign flip.

- **Boundary Handling**: Our functions handle boundaries with explicit zero-padding, which is equivalent to SciPy's `mode='valid'` when `padding=0`. SciPy's mode parameter provides a more abstract way to manage output size, where `mode='same'` means sufficient zero-padding is added so the output size matches the input size, and `mode='full'` means padding is added such that every part of the kernel overlaps with the image at least once.

- **Runtime Performance**: We timed all three implementations on the input image with various kernels.

| Kernel | Four Loops (s) | Two Loops (s) | Scipy (s) |
|--------|----------------|---------------|-----------|
| 9x9 Box Filter | 12.221 | 1.238 | 0.050 |
| 1x3 Dx Operator | 0.581 | 1.204 | 0.005 |
| 3x1 Dy Operator | 0.752 | 1.232 | 0.006 |

- **Naive four-loop implementation**: The slowest, as expected in general. Due to having four nested for loops, its performance degrades rapidly with increasing kernel size. However, for the smaller 1x3 kernels, this approach only performs three basic arithmetic operations, making it faster than the two-loop implementation.

- **Two-loop implementation**: For larger kernels like the 9x9 box filter, it is much faster than the four-loop approach. However, for the very small 1x3 and 3x1 kernels, this version is slower. For a tiny 1x3 kernel, the cost of creating slice objects (at `patch = padded_image[...]`) outweighs the benefit of vectorizing just three multiplications.

- **`scipy.signal.convolve2d` function**: Fastest by a wide margin.

### Visual Results

The results from our implementation and SciPy's are shown below. The source image is a picture of Rick Astley, who is both our favorite vocal artist and a prime example for demostrating edge detection, as his suit displays sharp contrast both horizontally and vertically. All results are visually indistinguishable apart from the slight brightness deviation in the Dx and Dy images, which resulted from us taking a sign flip.

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part1_1.png" alt="Figure 1" style="max-width: 100%; height: auto;">
  </div>
</div>

## Part 1.2: Edge Detection with Finite Differences

We achieve edge detection by approximating the gradient of an image, which indicates the direction and magnitude of the greatest intensity change. The process involves convolving the image with finite difference operators to find partial derivatives, combining them to compute the gradient magnitude, and thresholding this magnitude image to produce a binary edge map.

### Producing an Edge Map

We can approximate the gradient image by convolving the image with finite difference operators from the previous part. We approximate the horizontal derivative ($I_x$) by convolving with $D_x$ and the vertical derivative ($I_y$) by convolving with $D_y$ to highlight the vertical and horizontal edges. To get a single value representing the total strength of the gradient at each pixel, we compute the gradient magnitude, $G$, as the Euclidean norm of the gradient vector. The resulting gradient magnitude image will have high intensity values where the image changes sharply (i.e., at edges) and low intensity values in smooth, uniform regions.

To create a distinct edge map, the gradient magnitude image is binarized by selecting a threshold value, $T$. Each pixel in the gradient magnitude image is then classified as either an edge or not based on this threshold:

$$
\text{EdgePixel}(x, y) = \begin{cases} 255 \text{ (white)} & \text{if } G(x, y) > T \\ 0 \text{ (black)} & \text{if } G(x, y) \le T \end{cases}
$$

### Results and Analysis

The method is applied to the cameraman image. The results for several values of $T$ are shown below.

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part1_2.png" alt="Figure 2" style="max-width: 100%; height: auto;">
  </div>
</div>

The image shows eight binarized results using thresholds ranging from T=15.8 to T=78.6. The choice of an optimal threshold involves a qualitative trade-off:

- A low threshold is highly sensitive and captures fine details, including faint edges on the cameraman's face and texture in the grass. However, this sensitivity also leads to a significant amount of background noise.
- A high threshold is very effective at noise suppression, producing a clean, sparse edge map. However, it eliminates weaker edges, causing fragmentation and loss of continuity in the main objects, such as the gaps that appear in the tripod's legs.

A threshold of **$T=51.7$** appears to offer the best balance. At this level, the primary contours of the cameraman, his camera, and the tripod are well-defined and mostly continuous. At the same time, a significant amount of the noise present at lower thresholds has been eliminated, particularly in the sky and on the building.

## Part 1.3: Derivative of Gaussian (DoG) Filter for Edge Detection

In Part 1.2, using finite difference operators directly on the image produced effective but noisy edge maps. This section improves the edge maps by incorporating smoothing to reduce noise before differentiation. We convolve the image with a Gaussian filter before applying the difference operators to create a single, more efficient Derivative of Gaussian (DoG) filter that simultaneously performs smoothing and differentiation.

### Smoothing with a DoG Filter

We first smooth the image by convolving the original image, $I$, is convolved with a 9x9 2D Gaussian kernel, $G$, to produce a blurred image: $I_{smoothed} = I * G$. We then calculate the partial derivatives of the image by convolving the smoothed images with the finite difference operators, $D_x$ and $D_y$: $I_x = I_{smoothed} * D_x$ and $I_y = I_{smoothed} * D_y$. Since Convolution is associative, we may combine the two steps from the previous method into one:

$$(I * G) * D_x = I * (G * D_x) $$

We can then pre-compute the convolution of the Gaussian filter and the difference operators to create two new filters, known as Derivative of Gaussian (DoG) filters:

-   $DoG_x = G * D_x$
-   $DoG_y = G * D_y$

Convolving the original image $I$ with the DoG filters yields the same smoothed partial derivatives as the two-step process. After obtaining the smoothed derivatives $I_x$ and $I_y$ (with either method), the final steps are identical to Part 1.2.

### DoG Filter Visualization

The Gaussian filter and the resulting DoG filters are visualized as images.

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part1_3_filters.png" alt="Figure 4" style="max-width: 100%; height: auto;">
  </div>
</div>

The Gaussian filter is a bright spot in the center that smoothly fades to black, representing the weights used for blurring. The DoGx filter appears as a bright region on the left and a dark region on the right, which computes a smoothed difference in the horizontal direction to detect vertical edges. The DoGy filter appears as a bright region on the top and a dark region on the bottom, which computes a smoothed difference in the vertical direction to detect horizontal edges.

### Edge Detection Results
The DoG filters are applied to the cameraman image, and the results are shown below.

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part1_3.png" alt="Figure 3" style="max-width: 100%; height: auto;">
  </div>
</div>

-   **Comparison with Part 1.2**: The most obvious improvement is the significant reduction in noise. The `Smoothed (Gaussian)` image is visibly blurred, which suppresses fine textures and noise. This leads to significantly cleaner binary edge maps. The cleaner gradient magnitude allows for more effective thresholding. A threshold of **T=19.7** offers a good balance, as it generates a clean and continuous outline of the cameraman, camera, and tripod and effectively minimizes background noise.

## Part 2.1: Image Sharpening with the Unsharp Mask Filter

We implement image sharpening to increase the clarity of details and edges within an image. In this part, we implement the "unsharp mask" filter, an effective method for image sharpening which amplifies an image's high-frequency components.

### The Unsharp Masking Algorithm
The core of the unsharp masking algorithm is that if a blurred image represents the low frequencies, then subtracting this blurred version from the original image will isolate the high frequencies.

$$ \text{Image}_{\text{high}} = \text{Image}_{\text{orig}} - \text{Image}_{\text{low}} = \text{Image}_{\text{orig}} - (\text{Image}_{\text{orig}} * G) $$

To create a sharper image, we simply add these extracted high frequencies back to the original image. To control the intensity of the effect, the high-frequency component is scaled by a factor $ \alpha$.

$$ \text{Image}_{\text{sharpened}} = \text{Image}_{\text{orig}} + \alpha \cdot \text{Image}_{\text{high}} $$

After plugging-in and rearranging, we get:

$$ \text{Image}_{\text{sharpened}} = (1 + \alpha)\text{Image}_{\text{orig}} - \alpha(\text{Image}_{\text{orig}} * G) $$

To combine the process can be combined into a single kernel operation, we express the original image as a convolution with an identity kernel, $ \delta$ (a matrix of zeros with a single 1 at its center).

$$ \text{Image}_{\text{sharpened}} = (1 + \alpha)(\text{Image}_{\text{orig}} * \delta) - \alpha(\text{Image}_{\text{orig}} * G) $$

We then get the expression for a single unsharp mask filter, $H$:

$$ \text{Image}_{\text{sharpened}} = \text{Image}_{\text{orig}} * \underbrace{[(1 + \alpha)\delta - \alpha G]}_{H \text{ (Unsharp Mask Filter)}} $$

Convolving the original image with $H$ performs the sharpening operation in one pass.

### Sharpening Results (Taj Mahal)

We applied the unsharp mask filter to a photograph of the Taj Mahal, which is somewhat clear already.

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part2_1_taj2.png" alt="Figure 6" style="max-width: 100%; height: auto;">
  </div>
</div>

The top row shows the results of sharpening the original image. At moderate strengths ($α=1$, $α=3$), the filter successfully enhancing the architectural details, producing a visually clear image. However, when the strength is pushed to α=10, the image becomes oversharpened, making it appear grainy and harsh. Most prominently, bright and dark outlines appear around high-contrast edges, like the texture of the dome. The top row shows the results of sharpening the smoothed image. While sharpening does improve its perceived clarity (mostly around $α=3$), it fails to restore the original's quality.

We then apply the filter to a screenshot from the *Never Gonna Give You Up* MV. Rick simply can't stop his Rickrolling, so the image is blurry.

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part2_1_rick2.png" alt="Figure 5" style="max-width: 100%; height: auto;">
  </div>
</div>

Similar to the blurred Taj image, sharpening improves the perceived sharpness of the blurry source but does not portray the level of clarity in the static image of Rick (as in Part 1.1). Since the image is already blurred, blurring it further does not yield significantly different results. 

## Part 2.2: Hybrid Images

In this part, we create of hybrid images, which are perceived differently depending on the viewing distance. Our eyes perceive high-frequency details (sharp edges, textures) when an object is close, but these details blur out from a distance, leaving only the low-frequency information (overall shape, broad tones) visible. By blending the high frequencies of one image (Image A) with the low frequencies of another (Image B), we can create a single composite image that changes its interpretation as the viewer moves closer or further away.

### Frequency Separation and Combination

To extract the low-frequency component, we use a Gaussian filter, We generate a Gaussian kernel, $G_{low}$, with a standard deviation $\sigma_{low}$, where a larger sigma value corresponds to more aggressive blurring and a lower cutoff frequency. We then convolve Image B with this filter to obtain a low-frequency image, which contains only the broad shapes and tones of Image B and will form the structure of the hybrid image that is visible from a distance.

To extract the high-frequency details from  Image A, we use the same subtractive logic as the unsharp mask. We first compute a low-frequency version of Image A using a separate Gaussian filter, $G_{high}$, with another standard deviation, $\sigma_{high}$. The resulting high--frequency image is the original Image A minus its blurred version (the convolution of the original Image A and $G_{high}$) and contains the edges, lines, and textures from Image A that will be visible up close.

The final hybrid image is a simple summation of the high-frequency component from Image A and the low-frequency component from Image B.

### Detailed Analysis: Nutmeg & Derek

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part2_2_nutmeg_on_DerekPicture_comparison.png" alt="Figure 9" style="max-width: 100%; height: auto;">
  </div>
</div>

This example combines the high-frequency texture of a cat with the low-frequency structure of a human face. For this pair, we used $ \sigma_{high}=5.0$ and $ \sigma_{low}=5.0$. These cutoff values are chosen experimentally to ensure that the facial features of Nutmeg the Cat are sharp enough to be seen up close but subtle enough to fade from a distance, revealing Derek's face.

We plot the log magnitude of the 2D Fast Fourier Transform (FFT) to observe the image's energy distribution in the frequency domain. In these plots, the center corresponds to low frequencies, while points further from the center represent higher frequencies.

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part2_2_nutmeg_on_DerekPicture_freq_analysis.png" alt="Figure 9" style="max-width: 100%; height: auto;">
  </div>
</div>

The bottom row provides a clear visualization of the filtering in the frequency domain.
    - **FFT Image 1 & 2:** The FFTs of the original images show a typical distribution for natural images.
    - **FFT High-Freq:** As the low frequencies have been removed, the FFT of the high-pass filtered Nutmeg image brightens up further from the center.
    - **FFT Low-Freq:** As the high frequencies have been removed, the FFT of the low-pass filtered Derek image shows energy only in a bright cross-shape at the very center.
    - **FFT Hybrid:** The FFT of the final hybrid image is a composite of the two filtered transforms, containing both the low-frequency energy from Derek and the high-frequency energy from Nutmeg.

### Additional Examples: Idols
Below are two members of the Japanese idol group Takane no Nadeshiko, Mikuru Hoshitani and Momona Matsumoto. [Here] is the MV of the group's hit song, in which they co-lead. They are both beautiful girls, but their facial features and hairstyles are quite different.

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part2_2_mikuru_on_momona_comparison.png" alt="Figure 10" style="max-width: 100%; height: auto;">
  </div>
</div>

We produce two hybrid images, one with Mikuru as high-frequency and Momona as low-frequency and another the other way around. From afar, only the general silhouettes of the two girls' faces, such as Momona's signature twin-tail, as visible. Up close, distinct facial features, such as Mikuru's iconic big smile, become perceptible.



### Additional Examples: F1 Cars

Mercedes and Red Bull Racing were title rivals in 2021, but they both struggle now in 2025. Below are their 2025 cars, the Mercedes W16 and the Red Bull RB21.

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part2_2_rb21_on_w16_comparison.png" alt="Figure 11" style="max-width: 100%; height: auto;">
  </div>
</div>

We produce two hybrid images, one with the Red Bull as high-frequency and the Mercedes as low-frequency and another the other way around. From a distance, the two cars look perfectly normal. When viewed up close, we see the strange sight of a Mercedes-colored car sponsored by Oracle and with a massive Red Bull logo, as well as a Red-Bull-Colored car with a Petronas logo and their classic cyan-coloed stripe.

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/sauber_petronas.jpg" alt="Figure 11" style="max-width: 40%; height: auto;">
    <p><strong>Flashbacks from 1996-2005, when Red Bull holds a majority share in and Petronas title-sponsors the Sauber F1 Team.</strong></p>
  </div>
</div>

## Parts 2.3 & 2.4: Multiresolution Blending

Multiresolution blending elegantly stitches images together by creating a seamless transition. The core idea is to decompose two images into different frequency bands and blend them at each band separately, using a smoother transition for the lower frequencies and a sharper transition for the higher frequencies.

### Gaussian and Laplacian Stacks

The Gaussian stack creates progressively blurred versions of an image, representing its low-frequency content at different scales. The process is as follows:
-   The first level of the stack, $G_0$, is the original image itself.
-   Each subsequent level, $G_{i+1}$, is created by convolving the previous level, $G_i$, with a 2D Gaussian kernel.

The Laplacian stack stores the details that are present in each Gaussian stack but are blurred out and lost in the next Gaussian stack. It is derived from the Gaussian stack.
-   Each level of the Laplacian stack, $L_i$, is the difference between two adjacent levels of the Gaussian stack: $L_i = G_i - G_{i+1}$.
-   The very last level of the Laplacian stack, $L_N$, is simply the most blurred image from the Gaussian stack, $G_N$.
    
### The Blending Algorithm

We perform multiresolution blending using the following algorithm:

1. For the two input images to be blended, Image A and Image B, we compute their respective Laplacian stacks, $L_A$ and $L_B$.
2. We either create or import a mask image, $M$. This mask is a binary image where one region is white (pixel value 1) and the other is black (pixel value 0) and defines which parts of the final image should come from Image A versus Image B.
3. We create a Gaussian stack, $G_M$, from the mask $M$. This results in a series of masks, one for each frequency level. The masks for the high-frequency levels are sharp, while the masks for the low-frequency levels are heavily blurred.
4. We construct a new, blended Laplacian stack, $L_{blend}$. At each level `i`, the corresponding Laplacian levels of the input images are combined using the blurred mask from the same level of the Gaussian stack:

$$L_{blend}[i] = L_A[i] \cdot (1 - G_M[i]) + L_B[i] \cdot G_M[i]$$

5. The final blended image is obtained by summing allhe levels of the blended Laplacian stack, $L_{blend}$.

### Detailed Analysis: The "Oraple"
This classic example from the original paper blends an apple and an orange. We attemp the blending and recreate Figure 3.42 from Szeliski's *Computer Vision: Algorithms and Applications*.

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part2_3_oraple_figure_3_42.png" alt="Figure 12" style="max-width: 70%; height: auto;">
  </div>
</div>

The figure illustrates how the blending occurs at different frequency bands. The top three rows show the blending at high-frequency (L0), medium-frequency (L2), and low-frequency (L4) bands, and the final row shows the low-frequency residual (the most blurred Gaussian level). In the top row (high frequencies), the transition is sharp, blending the fine texture of the orange skin with the apple's smooth surface. In the lower-frequency rows, the Gaussian-blurred mask creates an increasingly smooth and gradual transition for the underlying color and shape.

By collapsing these separately blended frequency bands, we get the smooth "oraple" image:

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part2_4_oraple.png" alt="Figure 14" style="max-width: 100%; height: auto;">
  </div>
</div>

### Additional Blending Example: BMW M4 (Vertical Seam)
Released in 2014, the F82 generation of the BMW M4 was widely popular. It successor, the G82 generation, attracted some initial controversy due to its overly aggressive front profile but went on to sell well. We merge the front ends of both generations to compare which one looks better. A vertical mask is used to blend the left half of the older F82 model with the right half of the newer G82 model. The multiresolution blending creates a smooth transition down the centerline of the cars, effectively hiding the rift between the different hood, grille, and bumper designs. Unfortunately, the front licence plates are positioned at different heights, and it is impossible to mitigate a break like that.

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part2_4_bmw.png" alt="Figure 13" style="max-width: 50%; height: auto;">
  </div>
</div>

### Additional Blending Examples: Elden Ring (Irregular Mask)
The TGA 2022 Game of the Year, Elden Ring, impresses gamers with both its battle mechanics and its spectacular art scenes. While the lands in the base game is governed by the sacred Elden Ring, the Realm of Shadows, introduced in a 2024 DLC, is not. However, we can bring the Elden ring to the Realm of Shadows with blending! For this example, we used an irregular mask to blend a base-game art featuring the Elden Ring in the sky with another art from the DLC. We employ a mask that covers the Elden Ring and a portion of the sky to extract the Ring and position it above the landmass of the Realm of Shadow. The Gaussian blurring of the irregular mask ensures the Ring and its atmospheric effects appear naturally embedded in the new background.

*Elden Ring, O Elden Ring. Beget Order most elegant, from my tender reverie..*

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px;">
  <div style="text-align: center;">
    <img src="/assets/images/proj2/part2_4_souls.png" alt="Figure 15" style="max-width: 80%; height: auto;">
  </div>
</div>

[Here]: https://www.youtube.com/watch?v=MPywGQPLJPo
