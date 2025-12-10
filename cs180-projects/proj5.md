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

# Project 5A: The Power of Diffusion Models

Due Date: Tuesday, November 25, 2025 at 11:59pm

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

* Low Frequency Prompt (Far away): *"A photo of children playing in a park"*
* High Frequency Prompt (Close up): *"A digital illustration of a futuristic city"*

When viewed from a distance (or by squinting/blurring the image), the image appears to be a two children playing in a park. When viewed normally (or zoomed in), the buildings that from the city skyline emerge, and the children's bodies begin to look like buildings as well.

<div align="center">
  <img src="/assets/images/proj5/hybrid_images/part19_hybrid_image.png" width="300px" />
  <br/>
  <em>Hybrid Image: City (Low Freq) / Playground (High Freq)</em>
</div>

# Project 5B: Flow Matching from Scratch

Due Date: Friday, December 12, 2025 at 11:59pm

## Part 1: Training a Single-Step Denoising UNet

We warm up by building a simple one-step denoiser. The goal is to train a neural network to map a noisy image $z$ back to its clean version $x$. This serves as a precursor to the more complex diffusion models, effectively teaching a network to "undo" a fixed amount of Gaussian noise.

### 1.1 Implementing the UNet

To perform the denoising task, we implemented a standard UNet architecture. The UNet is designed to capture features at different scales through a series of downsampling operations and then reconstruct the image details through upsampling, utilizing skip connections to preserve high-frequency information. Our implementation consists of several blocks:
* **Simple Operations:** We defined basic building blocks like `Conv` (convolution + batch norm + GELU), `DownConv` (strided convolution for downsampling), `UpConv` (transposed convolution for upsampling), and `Flatten`/`Unflatten` operations for the bottleneck.
* **Composed Blocks:**
    * `DownBlock`: Combines a `DownConv` with a standard `ConvBlock` to reduce spatial resolution while increasing channel depth.
    * `UpBlock`: Combines an `UpConv` with a `ConvBlock`. Crucially, this block handles the concatenation of skip connection features from the encoder path.
* **Unconditional UNet:** The full network follows an encoder-decoder structure:
    1.  **Encoder (Down Path):** The input image (MNIST digit) is processed through a series of `DownBlock`s, reducing spatial resolution (e.g., $28 \times 28 \to 14 \times 14 \to 7 \times 7$) while increasing feature channels.
    2.  **Bottleneck:** The features are flattened to a dense vector and then unflattened, forcing the model to capture a global compressed representation of the image content.
    3.  **Decoder (Up Path):** The representation is passed through `UpBlock`s. At each stage, feature maps from the corresponding encoder level are concatenated (skip connections), allowing the model to recover fine spatial details that would otherwise be lost.
    4.  **Output:** A final convolution maps the features back to the original image space (1 channel for MNIST).

### 1.2 Using the UNet to Train a Denoiser

We trained the UNet to solve the denoising problem by optimizing the Mean Squared Error (MSE) loss between the denoised output and the original clean image:

$$L = \mathbb{E}_{z,x} ||D_\theta(z) - x||^2$$

To create the training data, we artificially corrupted clean MNIST images $x$ by adding Gaussian noise:

$$z = x + \sigma \epsilon, \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$

We visualized this process by applying different noise levels ($\sigma$) to the training images. As $\sigma$ increases, the digits become increasingly indistinguishable from the background noise.

<div align="center">
  <img src="/assets/images/proj5b/part12_0_noise_levels.png" width="700px" />
  <br/>
  <em>Visualization of MNIST digits with increasing noise levels ($\sigma$ varies from 0.0 to 1.0)</em>
</div>

### 1.2.1 Training

We trained the model for 5 epochs with a fixed noise level of $\sigma = 0.5$. The model learned to estimate the original image given the noisy input. We used the Adam optimizer with a standard learning rate.

**Results:**
The training loss decreased rapidly within the first epoch and stabilized, indicating the model successfully learned the denoising task.

<div align="center">
  <img src="/assets/images/proj5b/part12_1_train_loss.png" width="700px" />
  <br/>
  <em>Training loss curve over 5 epochs</em>
</div>

We visualized the denoising results on the test set after the 1st and 5th epochs.
* **After Epoch 1:** The model can already recover the general shape of the digits, though the edges are slightly fuzzy and some background noise remains.
* **After Epoch 5:** The denoised images are significantly sharper. The model effectively removes the graininess of the $\sigma=0.5$ noise, producing clean digits that closely resemble the ground truth.

<div align="center">
  <img src="/assets/images/proj5b/part12_1_epoch_1.png" width="700px" />
  <img src="/assets/images/proj5b/part12_1_epoch_5.png" width="700px" />
  <br/>
  <em>Results after Epoch 1 and Epoch 5 (Top: Original, Middle: Noisy, Bottom: Denoised)</em>
</div>

### 1.2.2 Out-of-Distribution Testing

The model was trained exclusively with a noise level of $\sigma=0.5$. To test its robustness, we evaluated it on images with varying noise levels $\sigma \in [0.0, 1.0]$.

**Observations:**
* **Low Noise ($\sigma < 0.5$):** The model performs reasonably well but tends to over-smooth the images. Since it expects a certain amount of noise to be present, it interprets some of the actual digit details as noise and attempts to remove them.
* **Target Noise ($\sigma \approx 0.5$):** The model performs best here, as expected, cleanly removing the noise.
* **High Noise ($\sigma > 0.5$):** The performance degrades as the noise level increases. At $\sigma=1.0$, the input is extremely corrupted. The model struggles to identify the digit structure, often producing blurry blobs or failing to recover the digit entirely. This highlights the limitation of a single-step denoiser trained on a fixed noise level—it lacks the generalization capability to handle unseen noise distributions effectively.

<div align="center">
  <img src="/assets/images/proj5b/part12_2_sample.png" width="700px" />
  <br/>
  <em>Denoising performance on out-of-distribution noise levels (Varying $\sigma$)</em>
</div>

### 1.2.3 Denoising Pure Noise

To test if the model can generate images from scratch, we attempted to use it to "denoise" pure Gaussian noise ($\epsilon \sim \mathcal{N}(0, I)$). We trained a separate model specifically for this task, taking pure noise as input and attempting to map it to a clean MNIST digit.

**Results and Analysis:**
Visualizing the results after training, we observe that the model does *not* generate a clear, distinct digit. Instead, the output resembles a blurry, average shape—often looking like a ghostly "8" or a "0".

<div align="center">
  <img src="/assets/images/proj5b/part12_3_train_loss.png" width="700px" />
  <br/>
  <em>Training loss for pure noise denoising</em>
</div>

<div align="center">
  <img src="/assets/images/proj5b/part12_3_epoch_1.png" width="700px" />
  <img src="/assets/images/proj5b/part12_3_epoch_5.png" width="700px" />
  <br/>
  <em>Results of denoising pure noise. The outputs are the average of the dataset.</em>
</div>

**Why does this happen?**
This behavior occurs because we are training with an L2 (Mean Squared Error) loss function on a problem where the input (pure noise) has zero correlation with the target (a specific digit). Mathematically, minimizing the L2 loss $L = \mathbb{E}[||y - f(x)||^2]$ leads the model to predict the expected value of the target distribution, $\mathbb{E}[y|x]$.

Since the input $x$ is random noise and provides no information about which digit $y$ should be, the model minimizes loss by predicting the average of *all* MNIST digits. This "average digit" naturally looks like a superposition of common digit structures, explaining the blurry, generic shape we observe. This demonstrates why single-step denoising is insufficient for generative modeling and motivates the need for iterative diffusion models (Flow Matching) in Part 2.

## Part 2: Training a Flow Matching Model

In Part 1, we saw that single-step denoising with a simple MSE loss yields a blurry "average" image rather than a sharp, distinct sample. To generate high-quality images, we need an iterative process. In this section, we move from simple denoising to Flow Matching.

Instead of trying to jump from pure noise $x_0$ to a clean image $x_1$ in a single step, we define a continuous path (flow) between the two distributions. We train a UNet to predict the "velocity" (flow) of this path at any given time $t$, allowing us to gradually guide random noise toward a realistic image.

### 2.1 Adding Time Conditioning to UNet

To model the flow over time $t \in [0, 1]$, the neural network needs to know *which* point in time it is currently processing. A noisy image at $t=0.1$ (mostly noise) requires different processing than one at $t=0.9$ (mostly clean).

We modified the UNet from Part 1 to accept a scalar time input $t$.
1.  **Embedding:** We embed the scalar $t$ using a Fully-Connected Block (`FCBlock`) consisting of `Linear -> GELU -> Linear` layers.
2.  **Conditioning:** This time embedding is injected into the UNet blocks. We use the embedding to scale the feature maps in the `UpBlock`s and the final `Unflatten` block.
    * If $h$ is the feature map and $w_t$ is the time embedding vector, the conditioned feature is $h' = h \cdot w_t$ (channel-wise multiplication).

This allows the network to dynamically adapt its weights based on the noise level.

### 2.2 Training the Time-Conditioned UNet

We trained the model to predict the flow $v_t = x_1 - x_0$.
* **Data:** We sample a clean image $x_1$ from MNIST and a pure noise vector $x_0 \sim \mathcal{N}(0, I)$.
* **Interpolation:** We create a noisy intermediate image $x_t$ using linear interpolation:
    $$x_t = (1 - t)x_0 + t x_1$$
* **Objective:** We train the network $u_\theta$ to minimize the MSE between its prediction and the true flow direction:
    $$L = ||u_\theta(x_t, t) - (x_1 - x_0)||^2$$

We trained this time-conditioned model for 10 epochs using the Adam optimizer.

**Results:**
The training loss decreased steadily, indicating the model successfully learned to predict the flow vector.

<div align="center">
  <img src="/assets/images/proj5b/part22_train_loss.png" width="700px" />
  <br/>
  <em>Training loss for the Time-Conditioned UNet</em>
</div>

### 2.3 Sampling from the Time-Conditioned UNet

Once trained, we can generate images by solving the Ordinary Differential Equation (ODE) defined by the learned flow. We start with pure noise at $t=0$ and iteratively update the image using the Euler method (standard numerical integration) until we reach $t=1$.

We visualized the samples generated by the model after 1, 5, and 10 epochs of training.
* **Epoch 1:** The generated images are very rough; while some digit-like blobs appear, they lack structure.
* **Epoch 5:** Legible digits begin to emerge. The shapes are distinct, though some unconnected lines persist.
* **Epoch 10:** The model generates mostly clear, recognizable digits, although ambiguities exist. The flow matching process successfully transforms random noise into the data distribution.

<div align="center">
  <img src="/assets/images/proj5b/part22_epoch_1.png" width="700px" />
  <img src="/assets/images/proj5b/part22_epoch_5.png" width="700px" />
  <img src="/assets/images/proj5b/part22_epoch_10.png" width="700px" />
  <br/>
  <em>Results after Epoch 1, 5, and 10</em>
</div>

### 2.4 Adding Class-Conditioning to UNet

While the time-conditioned model generates realistic digits, we have no control over *which* digit it generates. To fix this, we added Class Conditioning.

We extended the architecture to accept a class label $c \in \{0, \dots, 9\}$.
1.  **Conditioning:** Similar to the time embedding, we project the one-hot encoded class label through an `FCBlock` and use it to modulate the UNet features (multiplied alongside the time embedding).
2.  **Dropout:** To ensure the model can still generate unconditionally (or robustly), we implemented label dropout. During training, we drop the class label (replace it with a zero vector) 10% of the time ($p_{uncond} = 0.1$).

### 2.5 Training the Class-Conditioned UNet

We trained this upgraded model using the same flow matching objective as before, but now providing the class label $c$ as input.

**Results:**
The training loss curve shows convergence similar to the time-only model.

<div align="center">
  <img src="/assets/images/proj5b/part25_train_loss.png" width="700px" />
  <br/>
  <em>Training loss for the Class-Conditioned UNet</em>
</div>

### 2.6 Sampling from the Class-Conditioned UNet

With the trained class-conditional model, we can now request specific digits. We also utilize **Classifier-Free Guidance (CFG)** to improve the quality of the samples.
$$\text{predicted\_flow} = u_{uncond} + \gamma (u_{cond} - u_{uncond})$$
We used a guidance scale of $\gamma = 5.0$.

We visualized the generated results for epochs 1, 5, and 10, generating 4 instances of each digit (0-9).
* **Epoch 1:** The model struggles to adhere to the class constraints; the digits are messy and often incorrect.
* **Epoch 5:** The class consistency improves significantly. The model reliably generates the requested digit, though some noise remains.
* **Epoch 10:** The results are excellent. The digits are sharp, diverse, and strictly follow the class labels.


<div align="center">
  <img src="/assets/images/proj5b/part25_epoch_1.png" width="700px" />
  <img src="/assets/images/proj5b/part25_epoch_5.png" width="700px" />
  <img src="/assets/images/proj5b/part25_epoch_10.png" width="700px" />
  <br/>
  <em>Results after Epoch 1, 5, and 10</em>
</div>
