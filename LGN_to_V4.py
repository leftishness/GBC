import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy.signal import convolve2d, fftconvolve  # Import optimized FFT-based convolution
import time  # For timing steps


# Define LGN filtering function
def lgn_filtering(image, sigma_center=2, sigma_surround=5):
    center = gaussian(image, sigma=sigma_center)
    surround = gaussian(image, sigma=sigma_surround)
    output = center - surround
    return output


# Define Gabor filter function
def gabor_filter(size, wavelength, orientation, sigma=2):
    start_time = time.time()
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)
    theta = np.deg2rad(orientation)
    X_theta = X * np.cos(theta) + Y * np.sin(theta)
    Y_theta = -X * np.sin(theta) + Y * np.cos(theta)
    gabor = np.exp(-(X_theta ** 2 + Y_theta ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * X_theta / wavelength)
    end_time = time.time()
    return gabor


# V1 simple cells processing with optimized convolution
def v1_simple_processing(image, orientations=[0, 45, 90, 135], wavelength=10):
    filtered_images = []
    for orientation in orientations:
        # Create Gabor filter
        gabor = gabor_filter(size=image.shape[0], wavelength=wavelength, orientation=orientation)
        start_time = time.time()
        # Perform convolution using fftconvolve for optimization
        filtered_image = fftconvolve(image, gabor, mode="same")
        end_time = time.time()
        filtered_images.append(filtered_image)
    return np.stack(filtered_images, axis=-1)


# V1 complex cells processing
def v1_complex_processing(simple_cell_outputs):
    output = np.sqrt(np.sum(simple_cell_outputs ** 2, axis=-1))
    return output


# Feedback modulation
def feedback_modulation(complex_cell_output, feedback_strength=0.1):
    feedback = gaussian(complex_cell_output, sigma=5)
    modulated_output = complex_cell_output + feedback_strength * feedback
    return modulated_output


# V2 processing (texture extraction)
def v2_processing(v1_output):
    texture_filter = np.array([[1, -1], [-1, 1]])
    output = fftconvolve(v1_output, texture_filter, mode="same")
    return output


# V4 processing (object-level abstraction)
def v4_processing(v2_output):
    output = gaussian(v2_output, sigma=3)
    return output


# Introduce noise or lesions
def perturb_signal(image, lesion_type=None, noise_level=0.1):
    if lesion_type == "loss":
        mask = np.random.choice([0, 1], size=image.shape, p=[0.5, 0.5])  # Randomly zero out half of the pixels
        perturbed_image = image * mask
    elif lesion_type == "delay":
        perturbed_image = gaussian(image, sigma=3)  # Simulate temporal blurring
    elif lesion_type == "noise":
        perturbed_image = random_noise(image, mode='gaussian', var=noise_level ** 2)
    else:
        perturbed_image = image
    return perturbed_image


# Hierarchical processing pipeline
def hierarchical_processing_pipeline(image, perturbation=None, noise_level=0.1):
    # Apply perturbation to input image
    perturbed_image = perturb_signal(image, lesion_type=perturbation, noise_level=noise_level)

    # LGN processing
    lgn_output = lgn_filtering(perturbed_image)

    # V1 processing
    simple_outputs = v1_simple_processing(lgn_output)
    complex_output = v1_complex_processing(simple_outputs)

    # Feedback modulation
    feedback_output = feedback_modulation(complex_output)

    # V2 processing
    v2_output = v2_processing(feedback_output)

    # V4 processing
    v4_output = v4_processing(v2_output)

    return perturbed_image, lgn_output, complex_output, feedback_output, v2_output, v4_output


# Load natural image
image = camera() / 255.0

# Run pipeline for different perturbations
perturbations = ["None", "loss", "delay", "noise"]
results = {}
for perturbation in perturbations:
    results[perturbation] = hierarchical_processing_pipeline(image, perturbation=perturbation)

# Plot results
fig, axes = plt.subplots(len(perturbations), 6, figsize=(20, 12))
for i, perturbation in enumerate(perturbations):
    perturbed_image, lgn_output, complex_output, feedback_output, v2_output, v4_output = results[perturbation]
    axes[i, 0].imshow(perturbed_image, cmap="gray")
    axes[i, 0].set_title(f"{perturbation} Input")
    axes[i, 1].imshow(lgn_output, cmap="gray")
    axes[i, 1].set_title("LGN Output")
    axes[i, 2].imshow(complex_output, cmap="gray")
    axes[i, 2].set_title("V1 Complex Output")
    axes[i, 3].imshow(feedback_output, cmap="gray")
    axes[i, 3].set_title("Feedback Modulated")
    axes[i, 4].imshow(v2_output, cmap="gray")
    axes[i, 4].set_title("V2 Output")
    axes[i, 5].imshow(v4_output, cmap="gray")
    axes[i, 5].set_title("V4 Output")
plt.tight_layout()
plt.show()
