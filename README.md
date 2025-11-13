# Generative-Models-for-Image-to-Image-Translation-and-Generation

This repository contains implementations of different deep generative models for both image generation and image-to-image translation tasks. The work explores Variational Autoencoders (VAE), multiple types of GANs, and translation frameworks such as Conditional GANs (cGAN) and CycleGAN.

All code is written inside a single Jupyter Notebook: main.ipynb, including preprocessing, model architectures, training, evaluation, visualizations, a Streamlit interface, and an ngrok cell for generating a public demo link.

## 📌Implemented Tasks

### 1. Signature Image Generation (VAE + Simple GAN)
VAE: Learns latent representations of signature images using convolutional encoders/decoders.

Simple GAN: Generates fake signatures from random noise.

Dataset: Grayscale signature images resized to 64×64.

Augmentation: Scaling, rotation, and noise addition to handle the small dataset size.

### 2. Custom GAN on CIFAR-10 (Cats & Dogs Only)
Generator: DCGAN-style network producing 32×32 RGB images.
Discriminator: Siamese-style, comparing real vs. generated images and returning a similarity score.
Dataset: CIFAR-10, filtered for cats and dogs, normalized to [-1, 1].

### 3. Conditional GAN (cGAN) – Sketch to Face
Generator takes a sketch + noise and outputs a realistic face.
Discriminator evaluates (sketch, face) pairs.
Dataset: Person Face Sketches dataset with paired sketch-photo images (64×64).

### 4. CycleGAN – Face ↔ Sketch Translation

Two Generators: Face → Sketch and Sketch → Face.
Two PatchGAN Discriminators for patch realism.
Uses adversarial + cycle consistency + identity losses.
Bidirectional translation achieved reliably with longer training.
Models are checkpointed after every epoch.

⚙️ Training Setup

Optimizer: Adam (lr = 0.0002, betas = (0.5, 0.999))
Augmentation: Scaling, rotation, noise injection (signatures).
Checkpoints: Saved every epoch.
Environment: Notebook tested on Kaggle T4 GPU.

📊 Results Summary

VAE & GAN:

VAE reconstructs signatures well.
GAN signatures improve visually with training.

Custom GAN (Cats & Dogs):

Siamese discriminator improves realism.
Cat/dog features emerge clearly after training.

cGAN:

Blurry faces at first, clearer with more epochs.

CycleGAN:

Reliable face ↔ sketch translation.
Better detail with longer training.

📂 Datasets Used
[Signature Verification Dataset]

ℹ️ Kaggle link: (https://www.kaggle.com/datasets/robinreni/signature-verification-dataset)

[CIFAR-10 — A classic dataset consisting of 60,000 32×32 color images in 10 classes (50,000 train + 10,000 test)]

ℹ️ Download & info: (https://www.cs.toronto.edu/~kriz/cifar.html)

[Person Face Sketches — Dataset of paired face photos and sketches (21,000+ images)]

ℹ️ Kaggle link: (https://www.kaggle.com/datasets/almightyj/person-face-sketches)


