# MRI Physics and K-Space: Background for Deep Learning Reconstruction

## 1. How MRI Works

Magnetic Resonance Imaging (MRI) works by exploiting nuclear magnetic resonance of hydrogen
protons (¹H) in biological tissue. Unlike CT or X-ray, MRI uses no ionizing radiation. Instead:

1. **Polarization**: A strong static magnetic field B₀ (1.5T–7T) aligns proton spins along z.
2. **Excitation**: A radiofrequency (RF) pulse tips magnetization into the transverse plane.
3. **Spatial encoding**: Gradient coils create spatially varying field perturbations, encoding
   position into the frequency and phase of the precessing magnetization.
4. **Signal detection**: RF receiver coils detect the oscillating electromagnetic signal
   from relaxing spins.

The key insight: **the MRI scanner does not directly measure the image**. Instead, it measures
samples of the 2D Fourier transform of the image — called **k-space**.

---

## 2. K-Space: The Fourier Domain of MRI

### 2.1 What Is K-Space?

K-space is the Fourier transform of the spatial image. If the tissue magnetization distribution
is `m(x, y)`, then the MRI signal at position `(k_x, k_y)` in k-space is:

```
s(k_x, k_y) = ∫∫ m(x, y) · e^{-j2π(k_x·x + k_y·y)} dx dy
```

This is simply the 2D Fourier transform: `s = F{m}`.

To reconstruct the image from k-space measurements, we apply the inverse FFT:
```
m(x, y) = F⁻¹{s(k_x, k_y)}
```

### 2.2 K-Space Structure

K-space has a specific structure that is crucial for understanding MRI acceleration:

```
k-space layout (center = DC, high freq at periphery):

         ky
         │ ← Low spatial frequency (tissue contrast, large structures)
         │
─────────┼───────── kx
         │
         │ ← High spatial frequency (edges, fine detail)

Center:  Contains most of the image energy — tissue contrast, large structures
Edges:   High spatial frequencies — edges, small features, fine texture
```

**K-space energy distribution:**
- ~90% of total energy concentrated in central 10% of k-space
- Outer k-space contains critical edge/boundary information
- Missing outer k-space → blurred image (loss of resolution)
- Missing center k-space → loss of contrast, ghosting

### 2.3 K-Space Sampling in Clinical MRI

In standard clinical MRI, k-space is sampled line by line (one phase-encode line per TR):

```
Clinical 2D Cartesian acquisition:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Read direction (kx): sampled rapidly during gradient readout (~few ms)
Phase direction (ky): one line acquired per repetition time (TR)

Total scan time = (number of ky lines) × TR × (number of slices) × NEX
               = 256 × 1000ms × 20 × 1 ≈ 85 minutes (T1 brain)
```

For a typical 256×256 image: **256 repetitions needed** before a single image is formed.
This is why MRI is slow — unlike a camera, you cannot take a single "snapshot."

### 2.4 The Nyquist-Shannon Sampling Theorem

Full k-space sampling obeys the Nyquist theorem: we must sample at twice the highest
spatial frequency to avoid aliasing. The sampling density determines the field of view (FOV):

```
FOV_x = 1 / Δk_x        (readout direction)
FOV_y = 1 / Δk_y        (phase-encode direction)
Resolution = 1 / (N · Δk)
```

If we acquire fewer phase-encode lines than the Nyquist requirement, we **undersample**
k-space — and the image reconstructed by naive IFFT will show **aliasing artifacts**.

---

## 3. Accelerated MRI: Undersampling K-Space

### 3.1 Why Undersample?

The clinical bottleneck in MRI speed is the phase-encode direction: each TR (500ms–3000ms)
acquires only one k-space line. **Acquiring fewer lines directly reduces scan time**:

```
Acceleration factor R = (full k-space lines) / (acquired lines)

R=2: 2× faster scan
R=4: 4× faster (e.g., 20 min → 5 min)
R=8: 8× faster (e.g., 20 min → 2.5 min)
```

However, naive undersampling creates **coherent aliasing** (overlapping copies of
the image shifted by FOV/R), which is diagnostic. High-quality reconstruction
requires sophisticated algorithms to "undo" the aliasing.

### 3.2 Incoherent Undersampling for Compressed Sensing

Key insight from compressed sensing: if we undersample with a **random** or
**variable-density** pattern (rather than uniform), aliasing becomes incoherent
(noise-like rather than structured). Sparsity-based reconstruction can then
separate signal from aliasing noise.

```
Uniform undersampling (bad):          Random undersampling (CS-compatible):
─ ─ ─ ─ ─ ─ ─ ─ ─ ─                 ─ ─ ─ ─ ─    ─ ─ ─   ─
─ ─ ─ ─ ─ ─ ─ ─ ─ ─                 ─   ─ ─ ─ ─ ─     ─ ─ ─
─ ─ ─ ─ ─ ─ ─ ─ ─ ─                 ─ ─     ─ ─   ─ ─ ─ ─ ─
Periodic aliasing, diagnostic        Incoherent noise, CS-recoverable
```

### 3.3 Undersampling Mask Types

**1D Random (most common for benchmarks):**
```
ky pattern: ■ □ ■ □ □ ■ □ ■ □ □ ■ ■ □ ■ □ □ ■ □ ■ □
            Center fully sampled (ACS region)
```
Used in most deep learning papers. Simple, effective for single-coil evaluation.

**Equispaced (GRAPPA-compatible):**
```
ky pattern: ■ □ □ □ ■ □ □ □ ■ □ □ □ ■ □ □ □ ■ □ □ □
            Regular R-fold decimation with random offset
```
Simulates clinical parallel imaging acquisition. Compatible with GRAPPA calibration.

**2D Variable-Density Poisson Disc (for 3D acquisitions):**
```
           kz
           ■ □ □ ■ □ ■ □ ■
         ■ □ ■ ■ □ ■ ■ □ ■ □
       □ ■ ■ ■ ■ ■ ■ ■ ■ □ ■ □
     ■ □ ■ ■ ■ ■ ■ ■ ■ ■ ■ □ ■ ■   ← Dense center
       □ ■ ■ ■ ■ ■ ■ ■ ■ □ ■ □
         ■ □ ■ ■ □ ■ ■ □ ■ □
           ■ □ □ ■ □ ■ □ ■       ky
```
Variable density: fully sampled center, sparse periphery. More realistic for
prospective 3D cardiac and DCE acquisitions.

### 3.4 Auto-Calibration Signal (ACS)

The center k-space region is always fully sampled (regardless of total acceleration).
This is the **Auto-Calibration Signal (ACS)** or **calibration region**:

```
Typical ACS: central 5-10% of k-space lines
Purpose:
  1. Estimate coil sensitivity maps (parallel imaging)
  2. Train GRAPPA kernel
  3. Provide low-frequency contrast information
  4. Condition deep learning reconstruction networks
```

---

## 4. MRI Reconstruction Approaches

### 4.1 Classical Compressed Sensing

The compressed sensing framework (Lustig et al. 2007) formulates reconstruction as:

```
minimize   λ_TV · TV(x) + λ_wav · ‖Ψx‖₁
subject to ‖MFx - y‖₂² ≤ ε
```

Where:
- `F` = 2D FFT (Fourier encoding)
- `M` = undersampling mask
- `y` = measured k-space data
- `TV(x)` = total variation (edge-sparsity prior)
- `Ψ` = wavelet transform (texture-sparsity prior)
- `ε` = noise level

**Pros:** Interpretable, no training data, deterministic, well-understood error bounds.
**Cons:** Slow (iterative), parameter-sensitive, blurs fine detail at high acceleration.

### 4.2 Parallel Imaging (SENSE, GRAPPA)

Modern clinical scanners use multiple receive coils (8–64 channels). Each coil has a
spatial sensitivity profile `S_i(x, y)`. The measured signal for coil i is:

```
y_i = M · F · (S_i ⊙ x) + ε_i
```

**SENSE** (Pruessmann 1999): Solve for x using known sensitivity maps:
```
x = (S^H E^H E S)^{-1} S^H E^H y
```
Optimal SNR, requires accurate sensitivity maps.

**GRAPPA** (Griswold 2002): Data-driven k-space interpolation using local neighborhood.
Calibrates a kernel from ACS region, applies to entire k-space.
More robust to motion, widely used clinically.

### 4.3 Deep Learning Reconstruction

Deep learning replaces the hand-crafted CS prior with a learned prior:

```
x̂ = argmin   ½‖Ax - y‖² + λ·R_θ(x)
       x
```

Where R_θ is a neural network trained to penalize non-natural MRI images.

**U-Net (Supervised regression):**
```
x̂ = U-Net_θ(x_ZF)     x_ZF = F^{-1}(y)  (zero-filled)
Trained with L2 or L1 loss on ground truth
```
Fast, simple, but tends to over-smooth fine detail.

**Cascaded U-Net (E2E-VarNet):**
```
x₀ = x_ZF
x_{k+1} = U-Net_k(x_k) + data_consistency_k(x_k)
x̂ = x_K
```
Multi-stage refinement. Current SOTA for supervised methods.
Requires large labeled datasets; hallucinations rare but possible.

### 4.4 Score-Based Diffusion Models (Our Approach)

Instead of learning a regression mapping, we learn the **score function** of the
image distribution — the gradient of the log-density:

```
s_θ(x, σ) ≈ ∇_x log p_σ(x)
```

This score guides denoising from noise → image via Langevin dynamics or SDE solvers.
For MRI reconstruction, we condition on y and enforce data consistency:

```
Reverse SDE: dx = [f(x,t) - g(t)² · ∇_x log p_t(x|y)] dt + g(t) dW̄
Data consistency: x ← x - λ · F†M(MFx - y)     (at each step)
```

**Why diffusion is particularly suited for MRI:**

1. **Principled uncertainty quantification**: Sample multiple times to get error bars.
   ```
   x₁, x₂, ..., x_N ~ p(x|y)
   uncertainty = std(x₁, ..., x_N)
   ```

2. **High acceleration resilience**: At R=16x, only 6.25% of k-space is measured.
   A strong image prior (learned from thousands of MRI volumes) is critical.
   Diffusion models encode the full image distribution, not just an average.

3. **Hallucination detectability**: Multiple draws from the posterior show disagreement
   at hallucinated regions — a built-in consistency check unavailable to deterministic methods.

4. **No additional training for new anatomies**: The score function works with any mask/anatomy
   at test time. Supervised U-Nets require retraining for new acceleration factors.

5. **Posterior mean vs. MAP**: Diffusion samples from the full posterior p(x|y).
   MMSE (minimum mean square error) reconstruction = E[x|y] ≈ average over N samples.
   This is theoretically optimal for MSE — U-Nets approximate this with a single forward pass.

---

## 5. Fourier Transform Details for MRI

### 5.1 Centered 2D FFT Convention

MRI software uses a **centered** k-space representation where the DC component
(zero frequency, corresponding to mean image intensity) is at the center.

```python
# Centered 2D FFT (image domain → k-space)
def fft2c(x):
    return fftshift(fft2(ifftshift(x), norm="ortho"))

# Centered 2D IFFT (k-space → image domain)
def ifft2c(x):
    return fftshift(ifft2(ifftshift(x), norm="ortho"))
```

The `ifftshift` before FFT and `fftshift` after puts DC at center.
`norm="ortho"` ensures energy preservation: ‖Fx‖₂ = ‖x‖₂.

### 5.2 MRI Sampling Theorem in Practice

For a 2D spin-echo sequence (most common structural imaging):
```
kx: sampled continuously during readout gradient
    Δkx = γ/(2π) · Gx · Δt   (Hz/m → m⁻¹)
ky: stepped discretely per TR
    Δky = γ/(2π) · Gy · τy   (phase encode gradient × duration)

Image reconstruction: IFFT(kspace) → complex image
Magnitude image: |complex| (eliminates phase errors)
```

### 5.3 Root-Sum-of-Squares (RSS) Combination

For multi-coil MRI, each coil produces a separate complex image. RSS combination:

```
x_RSS(r) = √( Σᵢ |x_i(r)|² )
```

RSS is SNR-optimal when coil sensitivity maps are unknown. It is the standard ground
truth in the fastMRI challenge (all targets are RSS reconstructions from fully-sampled
multi-coil data).

---

## 6. Why Diffusion Models Are Suited for MRI — Technical Deep Dive

### 6.1 The Ill-Posed Nature of Accelerated MRI

At R=4x, 75% of k-space is missing. The system `y = MFx + ε` is highly underdetermined:
infinitely many x are consistent with measurements y. Any reconstruction method must
choose one x from this infinite set based on some prior.

**Prior types:**
- **CS:** x should be sparse in wavelet domain (explicit, interpretable)
- **U-Net:** x should look like supervised training pairs (implicit, data-driven)
- **Diffusion:** x should be drawn from p(x) = distribution of real MRI images (generative)

The generative prior is the richest — it captures the full statistical structure of
natural MRI images, including complex texture correlations that wavelet sparsity misses.

### 6.2 Posterior Sampling vs. MAP Estimation

Most reconstruction methods find the MAP estimate:
```
x̂_MAP = argmax p(x|y) = argmax p(y|x) · p(x)
```

Diffusion models sample from the posterior:
```
x̂ ~ p(x|y) ∝ p(y|x) · p(x)
```

The MAP estimate often corresponds to over-smoothed images (the "mean" of a multimodal
posterior can lie in a low-probability region). Posterior samples better reflect
the uncertainty and can capture fine detail that MAP misses.

### 6.3 Score Function Connection to Denoising

The score function has a beautiful connection to optimal denoising:

By **Tweedie's formula**, the optimal MMSE denoiser satisfies:
```
E[x₀ | x_t] = x_t + σ²(t) · ∇_x log p_σ(x_t)
             = x_t + σ²(t) · s_θ(x_t, t)
```

This means the score function encodes the "denoising direction" at every noise level.
Learning `s_θ` is equivalent to learning an infinite family of denoisers indexed by σ.
At inference, we iteratively apply these denoisers from high σ (pure noise) to low σ (clean image).

### 6.4 The Role of Data Consistency

Without data consistency, diffusion sampling generates unconditional samples from p(x):
random high-quality MRI images with no relationship to the actual patient.

Data consistency adds a gradient term pushing x toward the measured k-space data:
```
∇_x log p(y|x) = F†M(MFx - y) / σ²_noise
```

Combining prior and likelihood scores gives the posterior score:
```
∇_x log p(x|y) ≈ ∇_x log p_σ(x) + ∇_x log p(y|x)
```

This is the conditional diffusion framework that generates patient-specific
reconstructions consistent with both the measurements AND the prior.

---

## 7. Practical MRI Reconstruction Considerations

### 7.1 Phase Corrections

Real MRI data has spatially varying phase due to:
- B₀ field inhomogeneities (eddy currents, susceptibility)
- RF pulse imperfections
- Motion between phase-encode lines

For magnitude imaging (most clinical applications), phase is discarded after IFFT.
For phase-sensitive imaging (e.g., flow, B0 mapping), phase must be carefully unwrapped.

### 7.2 Noise Characteristics

MRI noise is:
- **Gaussian** in k-space (thermal Johnson noise from receive coil)
- **Rician** in magnitude images (|complex Gaussian| distribution)
- **Correlated** across coils (from mutual inductance)
- **Colored** after filtering and reconstruction

Deep learning models trained on magnitude images implicitly learn Rician noise statistics.
For low-SNR regimes, Rician bias correction may be needed.

### 7.3 Gibbs Ringing

Truncating k-space at finite spatial frequency creates ripples near sharp edges (Gibbs phenomenon):

```
k-space truncation → sinc point spread function → ringing near boundaries
```

Clinical mitigation: Hamming/Hanning windowing of k-space (slight resolution trade-off).
Deep learning: Models can learn to suppress ringing as part of reconstruction.

### 7.4 FOV Aliasing vs. k-Space Undersampling Aliasing

Two distinct aliasing types in MRI:
1. **FOV aliasing**: Object extends beyond field of view (wrap-around artifact)
   → Fixed by increasing FOV or applying anti-aliasing gradients
2. **Undersampling aliasing**: Insufficient phase-encode lines (coherent ghosting at R=2+)
   → Target for CS/DL reconstruction

---

## 8. Clinical MRI Applications and Acceleration Needs

### 8.1 Orthopedic (Knee, Shoulder, Hip)

- Standard protocol: T1, PD, T2, STIR sequences; ~4 sequences × ~15 min = 60 min total
- Clinical target: 4x acceleration → 15 min scan
- Key concerns: Cartilage, meniscus, ligament visualization; high spatial resolution needed
- fastMRI benchmark uses knee as primary evaluation anatomy

### 8.2 Neurological (Brain)

- Standard protocol: T1, T2, FLAIR, DWI, T1+Gd; ~6 sequences × ~5 min = 30 min
- Clinical target: 4-8x acceleration → 5-8 min total
- Key concerns: Lesion characterization, enhancement patterns, volumetric accuracy
- Multi-coil data (16-32 channels) enables higher acceleration

### 8.3 Cardiac

- Challenge: Heart motion during acquisition requires gating; ~15-20 min per exam
- Clinical target: 4-8x acceleration enables real-time imaging
- Key concerns: Ventricular function, wall motion, perfusion; temporal coherence critical
- 2D slices at multiple time points; diffusion must handle temporal dimension

### 8.4 Abdominal/Pelvic

- Challenge: Respiratory motion during scan; breath-hold limits acquisition time
- Clinical target: 8-16x acceleration for single breath-hold full liver coverage
- Radial trajectories common (more motion-robust than Cartesian)
- DL reconstruction particularly impactful here: enables diagnostic-quality free-breathing scans

### 8.5 Quantitative MRI

- T1 mapping, T2 mapping, MR fingerprinting: require many repeated measurements
- Clinical target: 8-16x acceleration; currently too slow for routine use
- DL reconstruction could enable clinical translation of quantitative biomarkers
- Particularly relevant for treatment response monitoring in oncology

---

## 9. The fastMRI Dataset

The fastMRI dataset (Zbontar et al. 2018) is the primary benchmark for accelerated
MRI reconstruction research. Released by NYU Langone Health and Meta AI Research.

### 9.1 Dataset Statistics

| Split | Knee Singlecoil | Knee Multicoil | Brain Multicoil |
|-------|----------------|----------------|----------------|
| Train | 973 volumes | 973 volumes | 4,469 volumes |
| Val | 199 volumes | 199 volumes | 1,130 volumes |
| Test | 118 volumes | 118 volumes | 1,371 volumes |

### 9.2 Data Format (HDF5)

Each .h5 file contains:
```
kspace: complex64 array, shape (num_slices, num_coils, height, width)
reconstruction_rss: float32 array, shape (num_slices, height, width)
    Root-sum-of-squares of fully-sampled multi-coil data
```

Volume-level attributes: acquisition type, patient ID, field strength.

### 9.3 Benchmark Metrics

The fastMRI challenge uses SSIM and PSNR on normalized magnitude images:
```python
# Standard fastMRI normalization
max_val = volume.max()
psnr = peak_signal_noise_ratio(pred / max_val, target / max_val)
ssim = structural_similarity(pred / max_val, target / max_val)
```

---

## 10. Future Directions

### 10.1 Real-Time Reconstruction

Goal: <100ms reconstruction for real-time guidance (intraoperative, intervention).
Approach: Consistency model distillation (1-2 step sampling) + on-device deployment.

### 10.2 Joint Acquisition-Reconstruction Optimization

Rather than designing the undersampling mask separately, jointly optimize mask and
reconstruction network end-to-end. Active learning approaches (greedy sequential
k-space selection) show promise for non-Cartesian trajectories.

### 10.3 Physics-Informed Diffusion

Incorporate explicit MRI physics models (Bloch equations, coil sensitivity maps,
B0/B1 field maps) into the diffusion prior and data consistency. This would enable
true multi-contrast and quantitative MRI reconstruction.

### 10.4 Uncertainty-Guided Clinical Decision Support

Use the uncertainty from multiple posterior samples to flag reconstructions for
radiologist review. High variance regions = uncertain reconstruction = potential
for missed findings. Directly actionable in clinical workflow.

---

*This document provides background for understanding MRI reconstruction in the context
of diffusion model development. For clinical questions, consult a board-certified radiologist
or MRI physicist.*
