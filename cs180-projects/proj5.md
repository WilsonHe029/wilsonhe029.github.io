---
title: Project 5
parent: CS 180
layout: default
nav_order: 8
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

# Project 5: Neural Radiance Field

Part 1 Due Date: Tuesday, November 25, 2025 at 11:59pm
Part 2 Due Date: Friday, December 12, 2025 at 11:59pm

## Part 0: Setup

In this section, we established the foundation for the project by setting up the DeepFloyd IF diffusion model. DeepFloyd is a state-of-the-art text-to-image diffusion model that operates in a two-stage process. For this part, we focused on the first stage, which generates base images at a $64 \times 64$ resolution.

We initialized the `IF-I-XL-v1.0` model pipeline and configured it to run on the GPU. To generate images, we passed text prompts into the stage 1 pipeline. We utilized a fixed manual seed (119104, the ASCII code of my initials) to ensure the random noise generation was deterministic, allowing for reproducible results. The inference process was set to run for 20 steps, during which the model iteratively denoised the latent variables conditioned on the text embeddings to produce the final output.

We tested the setup by generating images from three distinct prompts. The model successfully produced images corresponding to the descriptions below.

<div align="center">
  <table style="width=100%">
    <tr>
      <td align="center">
        <img src="/assets/images/proj5/part0_1_A_black_and_white_photo_of_a_woman_reading_a_book.png" width="200px" />
        <br/>
        <em>"A black and white photo of a woman reading a book"</em>
      </td>
      <td align="center">
        <img src="/assets/images/proj5/part0_2_An_acrylic_painting_of_a_tropical_beach_at_sunset.png" width="200px" />
        <br/>
        <em>"An acrylic painting of a tropical beach at sunset"</em>
      </td>
      <td align="center">
        <img src="/assets/images/proj5/part0_3_A_photo_of_abraham_lincoln.png" width="200px" />
        <br/>
        <em>"A photo of abraham lincoln"</em>
      </td>
    </tr>
  </table>
</div>

## Part 1: Sampling Loops

### 1.1 The Forward Process

In this section, we implemented the forward diffusion process, which gradually adds noise to a clean image. This process allows us to generate a noisy image $x_t$ at any specific timestep $t$ from the original clean image $x_0$. The forward process is modeled by taking a clean image and adding Gaussian noise scaled by coefficients determined by the timestep. We used the pre-computed `alphas_cumprod` array (denoted as $\bar{\alpha}_t$) from the DeepFloyd pipeline scheduler.

To generate a noisy image at timestep $t$, we applied the following equation:

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

In practice, this is implemented by sampling noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ and computing:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
$$

We scaled the original image (the Berkeley Campanile) to the range $[-1, 1]$ before applying noise, and rescaled back to $[0, 1]$ for visualization.

Below are the results of the forward process at timesteps $t = 250$, $t = 500$, and $t = 750$. As $t$ increases, the signal from the original image diminishes while the noise variance increases.

<div align="center">
  <table style="width=100%">
    <tr>
      <td align="center">
        <img src="/assets/images/proj5/campanile.jpg" width="200px" />
        <br/>
        <em>Original (Campanile)</em>
      </td>
      <td align="center">
        <img src="/assets/images/proj5/noisy_images/part11_t__250.png" width="200px" />
        <br/>
        <em>Noisy at t=250</em>
      </td>
      <td align="center">
        <img src="/assets/images/proj5/noisy_images/part11_t__500.png" width="200px" />
        <br/>
        <em>Noisy at t=500</em>
      </td>
      <td align="center">
        <img src="/assets/images/proj5/noisy_images/part11_t__750.png" width="200px" />
        <br/>
        <em>Noisy at t=750</em>
      </td>
    </tr>
  </table>
</div>

### 1.2 Classical Denoising

To establish a baseline for denoising performance, we attempted to remove the noise added in the forward process using classical signal processing techniques. specifically Gaussian blurring. We applied a Gaussian blur to the noisy images $x_t$ generated in Part 1.1 for timesteps $t = 250$, $500$, and $750$. The goal was to see if a standard low-pass filter could effectively separate the noise from the signal and recover the original image $x_0$. We used `torchvision.transforms.GaussianBlur` with a kernel size of 3 and a sigma chosen to smooth out the grain.

The results below show the noisy image alongside the Gaussian-blurred reconstruction for each timestep. As observed, while Gaussian blurring effectively removes some of the high-frequency noise, it does so at the cost of significant detail loss. The resulting images are blurry and fail to recover the structural integrity of the original Campanile, especially at higher noise levels ($t=500, 750$). This demonstrates the limitations of classical denoising methods for this task and motivates the need for the learned diffusion models we explore in the next sections.

<div align="center">
  <table style="width=100%">
    <tr>
      <th align="center">Timestep</th>
      <th align="center">Noisy Input ($x_t$)</th>
      <th align="center">Gaussian Blur Output</th>
    </tr>
    <tr>
      <td align="center"><strong>t = 250</strong></td>
      <td align="center"><img src="/assets/images/proj5/noise_vs_blur/part12_01_Noisy_t250.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/noise_vs_blur/part12_02_Blurred_t250.png" width="200px" /></td>
    </tr>
    <tr>
      <td align="center"><strong>t = 500</strong></td>
      <td align="center"><img src="/assets/images/proj5/noise_vs_blur/part12_03_Noisy_t500.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/noise_vs_blur/part12_04_Blurred_t500.png" width="200px" /></td>
    </tr>
    <tr>
      <td align="center"><strong>t = 750</strong></td>
      <td align="center"><img src="/assets/images/proj5/noise_vs_blur/part12_05_Noisy_t750.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/noise_vs_blur/part12_06_Blurred_t750.png" width="200px" /></td>
    </tr>
  </table>
</div>

### 1.3 One-Step Denoising

In this section, we utilized the pre-trained DeepFloyd UNet to perform "one-step denoising." Instead of iteratively removing noise, we attempted to recover the original clean image $x_0$ directly from a noisy image $x_t$ in a single step by estimating the noise component.

The UNet model is trained to predict the noise $\epsilon$ contained in a noisy image. By feeding the noisy image $x_t$ (generated in Part 1.1) and the corresponding timestep $t$ into the UNet, we obtained an estimate of the noise $\epsilon_\theta(x_t, t)$. To aid the model, we used the text prompt *"a high quality photo"*.

Using the noise estimate, we recovered the estimated original image $\hat{x}_0$ by rearranging the forward diffusion equation:

$$
\hat{x}_0 = \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

The results below compare the original image, the noisy input, and the UNet's estimated clean image. At $t=250$, the model recovers the original structure and details quite well, albeit with some blurring. As the noise level increases ($t=500$ and $t=750$), the recovery becomes more difficult; while the general shape of the Campanile is preserved, fine details are lost and new artifacts appear. This demonstrates that while the model has a strong prior for natural images, recovering fine details from high levels of noise in a single step is challenging.

<div align="center">
  <table style="width=100%">
    <tr>
      <th align="center">Original</th>
      <th align="center">Noisy Input ($x_t$)</th>
      <th align="center">Estimated Clean Image ($\hat{x}_0$)</th>
    </tr>
    <tr>
      <td align="center" colspan="3"><strong>Timestep t = 250</strong></td>
    </tr>
    <tr>
      <td align="center"><img src="/assets/images/proj5/one_step_denoising/t_250/part13_original.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/one_step_denoising/t_250/part13_noisy_t=250.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/one_step_denoising/t_250/part13_est._clean_t=250.png" width="200px" /></td>
    </tr>
     <tr>
      <td align="center" colspan="3"><strong>Timestep t = 500</strong></td>
    </tr>
    <tr>
      <td align="center"><img src="/assets/images/proj5/one_step_denoising/t_500/part13_est._clean_t=500.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/one_step_denoising/t_500/part13_noisy_t=500.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/one_step_denoising/t_500/part13_est._clean_t=500.png" width="200px" /></td>
    </tr>
     <tr>
      <td align="center" colspan="3"><strong>Timestep t = 750</strong></td>
    </tr>
    <tr>
      <td align="center"><img src="/assets/images/proj5/one_step_denoising/t_750/part13_est._clean_t=750.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/one_step_denoising/t_750/part13_noisy_t=750.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/one_step_denoising/t_750/part13_est._clean_t=750.png" width="200px" /></td>
    </tr>
  </table>
</div>

### 1.4 Iterative Denoising

In this section, we implemented the core of the diffusion process: iterative denoising. Instead of jumping directly from a noisy state to the clean image, the diffusion model is designed to iteratively denoise the image step-by-step. We implemented a sampling loop that starts with a highly noisy image at timestep $t$ and moves to a slightly less noisy image at timestep $t'$ (where $t' < t$).

To speed up the process, we used strided sampling. Instead of taking single steps (e.g., $990, 989, 988...$), we skip steps (e.g., $990, 960, 930...$) to reduce the computational cost while still maintaining generation quality.

For each step from $t$ to $t'$, we calculate the new image $x_{t'}$ using the following update rule (derived from the diffusion posterior):

$$
x_{t'} = \frac{\sqrt{\bar{\alpha}_{t'}}\beta_t}{1 - \bar{\alpha}_t} \hat{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t'})}{1 - \bar{\alpha}_t} x_t + v_\sigma
$$

Where:
* $x_t$ is the current noisy image.
* $\hat{x}_0$ is the clean image estimated by the UNet at the current step.
* $\alpha_t = \bar{\alpha}_t / \bar{\alpha}_{t'}$
* $\beta_t = 1 - \alpha_t$
* $v_\sigma$ is random noise added for stochasticity (though in this specific implementation, we often set variance to 0 for deterministic sampling or use `strided_timesteps` logic).

We started the process at $t_{start} = 990$ and ran the loop down to $t=0$. We compared the results of this iterative process against the One-Step Denoising method (applied to the same highly noisy input) and Gaussian Blurring. The Iterative Denoising result is significantly superior. It successfully reconstructs the fine details of the Campanile (the clock tower, the texture of the brick) and the surrounding trees. In contrast, the One-Step Denoising from $t=990$ produces a very rough approximation that captures the general silhouette but lacks all high-frequency detail. The Gaussian Blur fails completely at this noise level, resulting in a unrecognizable blob.

<div align="center">
  <table style="width=100%">
    <tr>
      <th align="center">Iterative Denoising</th>
      <th align="center">One-Step Denoising</th>
      <th align="center">Gaussian Blur</th>
    </tr>
    <tr>
      <td align="center"><img src="/assets/images/proj5/method_comparison/part14_iterative_denoised.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/method_comparison/part14_one_step_denoised.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/method_comparison/part14_gaussian_blurred.png" width="200px" /></td>
    </tr>
  </table>
</div>

### 1.5 Diffusion Model Sampling

In this section, we generated completely new images from scratch. To generate new images, we simply set the starting point $x_{T}$ (where $T=990$) to be pure random noise sampled from a standard Gaussian distribution: $x_{T} \sim \mathcal{N}(0, \mathbf{I})$. We then applied the same `iterative_denoise` function implemented in Part 1.4. We used the generic prompt *"a high quality photo"* to guide the generation process. Since the initial state is random noise, the model hallucinates a new image based on the learned priors in the UNet and the text conditioning.

We generated 5 different samples using this method. While the resolution is low ($64 \times 64$), the model successfully produces distinct, coherent structures resembling natural scenes or objects.

<div align="center">
  <table style="width=100%">
    <tr>
      <td align="center"><img src="/assets/images/proj5/generated_samples/part15_sample_01.png" width="150px" /></td>
      <td align="center"><img src="/assets/images/proj5/generated_samples/part15_sample_02.png" width="150px" /></td>
      <td align="center"><img src="/assets/images/proj5/generated_samples/part15_sample_03.png" width="150px" /></td>
      <td align="center"><img src="/assets/images/proj5/generated_samples/part15_sample_04.png" width="150px" /></td>
      <td align="center"><img src="/assets/images/proj5/generated_samples/part15_sample_05.png" width="150px" /></td>
    </tr>
    <tr>
      <td align="center"><em>Sample 1</em></td>
      <td align="center"><em>Sample 2</em></td>
      <td align="center"><em>Sample 3</em></td>
      <td align="center"><em>Sample 4</em></td>
      <td align="center"><em>Sample 5</em></td>
    </tr>
  </table>
</div>

### 1.6 Classifier-Free Guidance (CFG)

While the previous random samples were coherent, their quality can be significantly improved using Classifier-Free Guidance (CFG). In CFG, we compute two noise estimates at each timestep:

1.  $\epsilon_c$: The noise estimate conditioned on the text prompt (e.g., *"a high quality photo"*).
2.  $\epsilon_u$: The noise estimate conditioned on an empty prompt (`""`).

We then combine these estimates to form the final noise prediction $\epsilon$ using a guidance scale $\gamma$:

$$
\epsilon = \epsilon_u + \gamma (\epsilon_c - \epsilon_u)
$$

If $\gamma > 1$, the model is forced to prioritize the features described in the prompt more heavily. For this section, we used a guidance scale of $\gamma = 7$. We integrated this logic into our iterative denoising loop, computing both conditional and unconditional embeddings and noise estimates at every step.

We generated 5 samples using CFG. Compared to the results in Part 1.5 (without CFG), these images are noticeably sharper, contain more distinct objects, and have higher visual fidelity. The colors are more vibrant, and the structures are less "foggy," demonstrating the effectiveness of CFG in reducing mode collapse and improving image quality.

<div align="center">
  <table style="width=100%">
    <tr>
      <td align="center"><img src="/assets/images/proj5/cfg_generated_samples/part16_cfg_sample_01.png" width="150px" /></td>
      <td align="center"><img src="/assets/images/proj5/cfg_generated_samples/part16_cfg_sample_02.png" width="150px" /></td>
      <td align="center"><img src="/assets/images/proj5/cfg_generated_samples/part16_cfg_sample_03.png" width="150px" /></td>
      <td align="center"><img src="/assets/images/proj5/cfg_generated_samples/part16_cfg_sample_04.png" width="150px" /></td>
      <td align="center"><img src="/assets/images/proj5/cfg_generated_samples/part16_cfg_sample_05.png" width="150px" /></td>
    </tr>
    <tr>
      <td align="center"><em>CFG Sample 1</em></td>
      <td align="center"><em>CFG Sample 2</em></td>
      <td align="center"><em>CFG Sample 3</em></td>
      <td align="center"><em>CFG Sample 4</em></td>
      <td align="center"><em>CFG Sample 5</em></td>
    </tr>
  </table>
</div>

### 1.7: Image-to-Image Translation

In this section, we explore "Image-to-Image Translation" using the SDEdit algorithm. This method allows us to guide the generation process not just with text, but also with an initial image. The core idea of SDEdit is to run the forward diffusion process on an existing image $x_0$ to obtain a noisy version $x_t$ at a specific timestep $t$, and then run the reverse diffusion (denoising) process starting from this $x_t$ rather than from pure noise.

The "noise level" determines how much freedom the model has to deviate from the original image. We control this using a starting index `i_start` in our timestep scheduler:

* **Low `i_start`** (e.g., 1 or 3): Corresponds to a high timestep $t$ (closer to $T=990$). The image is heavily corrupted with noise, allowing the model to "hallucinate" significantly and generate new structures that follow the text prompt while loosely adhering to the original color/layout.
* **High `i_start`** (e.g., 10 or 20): Corresponds to a low timestep $t$. The image has little noise, so the model primarily performs "cleanup" or minor stylization, keeping the output strictly faithful to the original input.

### 1.7.1 Editing Hand-Drawn and Web Images

We applied this technique to two different types of inputs to demonstrate its versatility: a downloaded web image and a crude hand-drawn sketch.

**Web Image Input**

We used an image of an anime character (Sakiko Togawa from the *BanG Dream!* series) and applied SDEdit with varying starting indices.
* At i_start = 1, the model generates an image that follows the prompt and general color scheme but changes the character's identity and pose significantly.
* At i_start = 20, the output is very similar to the original both in composition and color.

<div align="center">
  <table style="width=100%">
    <tr>
      <td align="center" colspan="6">
        <img src="/assets/images/proj5/sakiko.png" width="200px" /><br/>
        <em>Original Image</em>
      </td>
    </tr>
    <tr>
      <td align="center"><img src="/assets/images/proj5/web_image_edits/part17_1_web_edit_i_start_01.png" width="100px" /><br/>i_start=1</td>
      <td align="center"><img src="/assets/images/proj5/web_image_edits/part17_1_web_edit_i_start_03.png" width="100px" /><br/>i_start=3</td>
      <td align="center"><img src="/assets/images/proj5/web_image_edits/part17_1_web_edit_i_start_05.png" width="100px" /><br/>i_start=5</td>
      <td align="center"><img src="/assets/images/proj5/web_image_edits/part17_1_web_edit_i_start_07.png" width="100px" /><br/>i_start=7</td>
      <td align="center"><img src="/assets/images/proj5/web_image_edits/part17_1_web_edit_i_start_10.png" width="100px" /><br/>i_start=10</td>
      <td align="center"><img src="/assets/images/proj5/web_image_edits/part17_1_web_edit_i_start_20.png" width="100px" /><br/>i_start=20</td>
    </tr>
  </table>
</div>

**Hand-Drawn Input**

We also tested the model on a hand-drawn sketch of another anime character (Hitori "Bocchi" Gotoh from *Bocchi the Rock!*).
* At low indices (1, 3), the model uses the sketch as a loose guide for composition and color, filling in the details to create a fully rendered 3D or photorealistic character.
* At high indices (20), the model interprets the pencil strokes as actual texture, preserving the "sketchy" look rather than converting it to a realistic image.

<div align="center">
  <table style="width=100%">
    <tr>
      <td align="center" colspan="6">
        <img src="/assets/images/proj5/bocchi.jpg" width="200px" /><br/>
        <em>Original Sketch</em>
      </td>
    </tr>
    <tr>
      <td align="center"><img src="/assets/images/proj5/drawn_image_edits/part17_1_drawn_edit_i_start_01.png" width="100px" /><br/>i_start=1</td>
      <td align="center"><img src="/assets/images/proj5/drawn_image_edits/part17_1_drawn_edit_i_start_03.png" width="100px" /><br/>i_start=3</td>
      <td align="center"><img src="/assets/images/proj5/drawn_image_edits/part17_1_drawn_edit_i_start_05.png" width="100px" /><br/>i_start=5</td>
      <td align="center"><img src="/assets/images/proj5/drawn_image_edits/part17_1_drawn_edit_i_start_07.png" width="100px" /><br/>i_start=7</td>
      <td align="center"><img src="/assets/images/proj5/drawn_image_edits/part17_1_drawn_edit_i_start_10.png" width="100px" /><br/>i_start=10</td>
      <td align="center"><img src="/assets/images/proj5/drawn_image_edits/part17_1_drawn_edit_i_start_20.png" width="100px" /><br/>i_start=20</td>
    </tr>
  </table>
</div>

### 1.7.2 Inpainting

In this section, we implemented Inpainting, a technique used to modify specific regions of an image while preserving the rest. We used the RePaint algorithm, which allows a standard diffusion model to perform inpainting without fine-tuning. The goal of inpainting is to generate new content in a specific area (defined by a binary mask) that is coherent with the surrounding "known" pixels.

To achieve this, we modified the sampling loop. At each timestep $t$ in the reverse process, we have a generated image $x_t$. However, we also have access to the ground truth original image $x_{orig}$. We can ensure the unmasked regions stay correct by forcing them to match a noisy version of the original image.

Specifically, at every step $t$, we compute the next state $x_{t-1}$ as follows:

1.  **Sample:** Compute $x_{t-1}^{sample}$ using the standard diffusion update rule (denoising).
2.  **Known:** Compute a noisy version of the original image $x_{t-1}^{known}$ by adding noise to $x_{orig}$ corresponding to timestep $t-1$.
3.  **Combine:** Merge the two using the mask $\mathbf{m}$ (where 1 represents "keep original" and 0 represents "inpaint"):
    $$
    x_{t-1} = \mathbf{m} \odot x_{t-1}^{known} + (1 - \mathbf{m}) \odot x_{t-1}^{sample}
    $$

This forces the model to generate content only inside the mask, while the boundary conditions provided by the "known" pixels ensure the generated content blends seamlessly with the original image.

**Campanile Inpainting**

We applied this method to the Campanile image, masking out the top of the tower to generate a new design.

<div align="center">
  <table style="width=100%">
    <tr>
      <th align="center">Original</th>
      <th align="center">Mask</th>
      <th align="center">Inpainted Result</th>
    </tr>
    <tr>
      <td align="center"><img src="/assets/images/proj5/inpainting_results/part17_2_campanile_original.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/inpainting_results/part17_2_campanile_mask.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/inpainting_results/part17_2_campanile_inpainted.png" width="200px" /></td>
    </tr>
  </table>
</div>

**Custom Image 1: Sad Dog**
<div align="center">
  <table style="width=100%">
    <tr>
      <th align="center">Original</th>
      <th align="center">Mask</th>
      <th align="center">Inpainted Result</th>
    </tr>
    <tr>
      <td align="center"><img src="/assets/images/proj5/inpainting_results/part17_2_custom_image_1_original.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/inpainting_results/part17_2_custom_image_1_mask.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/inpainting_results/part17_2_custom_image_1_inpainted.png" width="200px" /></td>
    </tr>
  </table>
</div>

**Custom Image 2: Haunted Cat**
<div align="center">
  <table style="width=100%">
    <tr>
      <th align="center">Original</th>
      <th align="center">Mask</th>
      <th align="center">Inpainted Result</th>
    </tr>
    <tr>
      <td align="center"><img src="/assets/images/proj5/inpainting_results/part17_2_custom_image_2_original.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/inpainting_results/part17_2_custom_image_2_mask.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/inpainting_results/part17_2_custom_image_2_inpainted.png" width="200px" /></td>
    </tr>
  </table>
</div>

### 1.7.3 Text-Conditioned Image-to-Image Translation

In this section, we continue using the SDEdit technique (adding noise and then denoising), but we specifically focus on changing the semantic content of the image using text prompts. The methodology remains the same as in Section 1.7.1: we add noise to an original image $x_0$ to reach state $x_t$ (controlled by `i_start`), and then denoise it using a specific text prompt.

The key observation here is the trade-off between fidelity to the original image and adherence to the new text prompt.
* **Low `i_start`** (High Noise): The model has more freedom to hallucinate. It can transform the object in the image into something entirely different (e.g., a tower into a rocket) because the structural constraints of the original image are weakened by the noise.
* **High `i_start`** (Low Noise): The model is constrained by the original pixels. Even with a prompt like "Rocket Ship," if the noise level is too low, the model will simply denoise it back to the original tower.

**Example 1: Campanile $\rightarrow$ Beach**

We attempted to turn the Campanile into a sandy beach
* At high noise levels (`i_start=1, 3`), the tower is successfully replaced by palm trees, and the sky changes color.
* At low noise levels (`i_start=10, 20`), the original tower persists, although a tropical-like atmosphere began to emerge.

<div align="center">
  <table style="width=100%">
    <tr>
      <td align="center"><img src="/assets/images/proj5/sdedit/Campanile/part17_3_i_start1_t960.png" width="100px" /><br/>i_start=1</td>
      <td align="center"><img src="/assets/images/proj5/sdedit/Campanile/part17_3_i_start3_t900.png" width="100px" /><br/>i_start=3</td>
      <td align="center"><img src="/assets/images/proj5/sdedit/Campanile/part17_3_i_start5_t840.png" width="100px" /><br/>i_start=5</td>
      <!-- <td align="center"><img src="/assets/images/proj5/sdedit/Campanile/part17_3_i_start7_t780.png" width="100px" /><br/>i_start=7</td> -->
      <td align="center"><img src="/assets/images/proj5/sdedit/Campanile/part17_3_i_start10_t690.png" width="100px" /><br/>i_start=10</td>
      <td align="center"><img src="/assets/images/proj5/sdedit/Campanile/part17_3_i_start20_t390.png" width="100px" /><br/>i_start=20</td>
      <td align="center"><img src="/assets/images/proj5/sdedit/Campanile/part17_3_Original.png" width="100px" /><br/>Original</td>
    </tr>
  </table>
</div>

**Example 2: Cat $\rightarrow$ Bear**

We applied a prompt to transform a photo of a cat into a bear fishing in a river.
<!-- * At `i_start=1`, the animal is completely transformed from a cat into a bear.
* As `i_start` increases, the original cat's features (such as pose) reappear. -->

<div align="center">
  <table style="width=100%">
    <tr>
      <td align="center"><img src="/assets/images/proj5/sdedit/Cat2Bear/part17_3_ii_start1_t960.png" width="100px" /><br/>i_start=1</td>
      <td align="center"><img src="/assets/images/proj5/sdedit/Cat2Bear/part17_3_ii_start3_t900.png" width="100px" /><br/>i_start=3</td>
      <td align="center"><img src="/assets/images/proj5/sdedit/Cat2Bear/part17_3_ii_start5_t840.png" width="100px" /><br/>i_start=5</td>
      <!-- <td align="center"><img src="/assets/images/proj5/sdedit/Cat2Bear/part17_3_ii_start7_t780.png" width="100px" /><br/>i_start=7</td> -->
      <td align="center"><img src="/assets/images/proj5/sdedit/Cat2Bear/part17_3_ii_start10_t690.png" width="100px" /><br/>i_start=10</td>
      <td align="center"><img src="/assets/images/proj5/sdedit/Cat2Bear/part17_3_ii_start20_t390.png" width="100px" /><br/>i_start=20</td>
      <td align="center"><img src="/assets/images/proj5/sdedit/Cat2Bear/part17_3_Original2.png" width="100px" /><br/>Original Image</td>
    </tr>
  </table>
</div>

**Example 3: Dog $\rightarrow$ Cat**

We transformed a photo of a dog into an artistic painting of a cat wearing sunglasses.
<!-- * At `i_start=1`, the image loses its photorealism and adopts the texture and strokes of the target style.
* At `i_start=20`, it remains a standard photo. -->

<div align="center">
  <table style="width=100%">
    <tr>
      <td align="center"><img src="/assets/images/proj5/sdedit/Dog2Cat/part17_3_iii_start1_t960.png" width="100px" /><br/>i_start=1</td>
      <td align="center"><img src="/assets/images/proj5/sdedit/Dog2Cat/part17_3_iii_start3_t900.png" width="100px" /><br/>i_start=3</td>
      <td align="center"><img src="/assets/images/proj5/sdedit/Dog2Cat/part17_3_iii_start5_t840.png" width="100px" /><br/>i_start=5</td>
      <!-- <td align="center"><img src="/assets/images/proj5/sdedit/Dog2Cat/part17_3_iii_start7_t780.png" width="100px" /><br/>i_start=7</td> -->
      <td align="center"><img src="/assets/images/proj5/sdedit/Dog2Cat/part17_3_iii_start10_t690.png" width="100px" /><br/>i_start=10</td>
      <td align="center"><img src="/assets/images/proj5/sdedit/Dog2Cat/part17_3_iii_start20_t390.png" width="100px" /><br/>i_start=20</td>
      <td align="center"><img src="/assets/images/proj5/sdedit/Dog2Cat/part17_3_Original3.png" width="100px" /><br/>Original Image</td>
    </tr>
  </table>
</div>

### 1.8 Visual Anagrams

In this section, we created "Visual Anagrams"—optical illusions where an image looks like one object when viewed upright, but looks like a completely different object when flipped upside down. To achieve this, we modified the iterative denoising process to satisfy two different text prompts simultaneously: one for the upright image and one for the flipped image.

At each timestep $t$, we computed the noise estimate $\epsilon$ by averaging the estimates from both prompts, but with a geometric transformation applied to the second one.
1.  **Upright Estimate:** We computed $\epsilon_1$ using the current image $x_t$ and the first prompt (e.g., *"an oil painting of an old man"*).
2.  **Flipped Estimate:** We flipped the image $x_t$ upside down to get $x_t'$, and computed $\epsilon_2$ using $x_t'$ and the second prompt (e.g., *"an oil painting of people around a campfire"*).
3.  **Combination:** We flipped $\epsilon_2$ back to the original orientation to get $\epsilon_2'$. The final noise estimate used for the diffusion step was the average:
    $$
    \epsilon_{total} = \frac{\epsilon_1 + \epsilon_2'}{2}
    $$

This forces the diffusion model to generate an image that aligns with both prompts in their respective orientations.

**Illusion 1: The steam castle**

* Upright: *"An engraving of a vintage steam train"*
* Flipped: *"A chalk drawing of a medieval castle"*

<div align="center">
  <table style="width=100%">
    <tr>
      <th align="center">Upright View</th>
      <th align="center">Flipped View</th>
    </tr>
    <tr>
      <td align="center"><img src="/assets/images/proj5/visual_anagrams/part18_illusion1_upright.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/visual_anagrams/part18_illusion1_flipped.png" width="200px" /></td>
    </tr>
  </table>
</div>

**Illusion 2: The Bearcat**

* Upright: *"A photo of a grizzly bear fishing in a river"*
* Flipped: *"A photo of a cat wearing sunglasses"*

<div align="center">
  <table style="width=100%">
    <tr>
      <th align="center">Upright View</th>
      <th align="center">Flipped View</th>
    </tr>
    <tr>
      <td align="center"><img src="/assets/images/proj5/visual_anagrams/part18_illusion2_upright.png" width="200px" /></td>
      <td align="center"><img src="/assets/images/proj5/visual_anagrams/part18_illusion2_flipped.png" width="200px" /></td>
    </tr>
  </table>
</div>

### 1.9 Hybrid Images

In this section, we created "Hybrid Images"—images that look like one thing from close up and another from far away. This effect, as outlined back in Project 2, relies on the fact that the human visual system perceives high frequencies (fine details) when close to an image, and low frequencies (broad shapes) when far away. We can generate such images using Factorized Diffusion. Similar to the Visual Anagrams, we computed two separate noise estimates at each timestep $t$, but combined them in the frequency domain:

1.  **$\epsilon_1$ (Low Frequencies):** Generated using prompt 1 (e.g., *"A digital illustration of a futuristic city"*).
2.  **$\epsilon_2$ (High Frequencies):** Generated using prompt 2 (e.g., *"A photo of children playing in a park"*).
3.  **Combination:** We applied a low-pass filter $f_{low}$ to $\epsilon_1$ and a high-pass filter $f_{high}$ to $\epsilon_2$, then summed them:
    $$
    \epsilon_{total} = f_{low}(\epsilon_1) + f_{high}(\epsilon_2)
    $$

The low-pass filter was implemented using a Gaussian blur with a large kernel (size 33, sigma 2), and the high-pass filter was simply the inverse (Original - LowPass).

**The City and Playground**

* Low Frequency Prompt (Far away): *"A digital illustration of a futuristic city"*
* High Frequency Prompt (Close up): *"A photo of children playing in a park"*

When viewed from a distance (or by squinting/blurring the image), the image appears to be a city skyline. When viewed normally (or zoomed in), the children and the green trees of the park emerge.

<div align="center">
  <img src="/assets/images/proj5/hybrid_images/part19_hybrid_image.png" width="300px" />
  <br/>
  <em>Hybrid Image: City (Low Freq) / Playground (High Freq)</em>
</div>
