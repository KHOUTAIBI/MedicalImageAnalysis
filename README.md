# Medical Image Analysis

> A research-oriented repository on **medical image analysis** and **representation learning**, combining  
> **geometry-aware variational autoencoders**, **CNN-based experiments**, and a notebook on **semantic manifolds**.  
> The project explores how latent geometric structure — in particular **spherical** and **toroidal** manifolds — can be used for representation learning and inference.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Mathematical Background](#3-mathematical-background)
   - 3.1 [Why Manifold-Constrained Latent Spaces?](#31-why-manifold-constrained-latent-spaces)
   - 3.2 [Variational Autoencoders](#32-variational-autoencoders)
   - 3.3 [Spherical and Toroidal Latent Spaces](#33-spherical-and-toroidal-latent-spaces)
   - 3.4 [Training Objective](#34-training-objective)
4. [Code Walkthrough](#4-code-walkthrough)
   - 4.1 [`main.py`](#41-mainpy)
   - 4.2 [`model/`](#42-model)
   - 4.3 [`dataloader/`](#43-dataloader)
   - 4.4 [`CNN_experiment/`](#44-cnn_experiment)
   - 4.5 [`Semantic Manifolds/`](#45-semantic-manifolds)
5. [Installation & Usage](#5-installation--usage)
6. [Notes](#6-notes)

---

## 1. Project Overview

This repository gathers several experiments in **medical image analysis** with a strong emphasis on **representation learning** and **latent geometry**.

At a high level, the project combines:

- **geometry-aware variational autoencoders**,
- **CNN-based experiments**,
- a notebook-based exploration of **semantic manifolds**,
- custom data loading and inference utilities.

A central idea of the repository is that Euclidean latent spaces are not always the most natural choice for structured data.  
For certain types of representations, it can be useful to constrain the latent code to live on a **sphere** or on a **torus**, leading to models such as:

- a **Spherical VAE**,
- a **Toroidal VAE**.

This is especially relevant in settings where the data may exhibit periodic structure, angular structure, or smooth low-dimensional organization in latent space.

---

## 2. Repository Structure

```text
MedicalImageAnalysis/
│
├── CNN_experiment/          # CNN-based experiment folder with training utilities and saved checkpoints
├── Semantic Manifolds/      # Notebook-based project on semantic manifolds / emotion classification
├── dataloader/              # Data loading utilities
├── model/                   # Core models: spherical VAE, toroidal VAE, training and inference code
├── .gitignore
├── README.md
└── main.py                  # Main entry point for training / inference
```

More specifically:

```text
CNN_experiment/
├── saves/                   # Saved model outputs / checkpoints
├── circular_VAE.py          # Circular / geometry-aware VAE variant
├── dataset.py               # Dataset definition for the experiment
├── errors.py                # Error / metric utilities
├── ive.py                   # Special mathematical utility
├── requirements.txt         # Local dependencies for the experiment
├── train_experiment.py      # Training script for the CNN experiment
└── von_mises_fisher.py      # vMF-related distribution utilities
```

```text
Semantic Manifolds/
└── emotion_classification.ipynb   # Notebook on semantic manifolds and emotion classification
```

```text
dataloader/
├── data.py                  # Data-loading entry points
└── utils.py                 # Data utilities and preprocessing helpers
```

```text
model/
├── infer.py                 # Inference utilities
├── spherical_vae.py         # Spherical VAE implementation
├── torus_vae.py             # Toroidal VAE implementation
└── train.py                 # Training loop
```

---

## 3. Mathematical Background

### 3.1 Why Manifold-Constrained Latent Spaces?

Standard latent-variable models often use a Euclidean latent space:

$$
z \in \mathbb{R}^d
$$

However, some data structures are naturally better represented on constrained manifolds.

Examples:

- **angular / directional data** are often better modeled on a sphere,
- **periodic variables** are naturally related to circular or toroidal geometry,
- structured semantic factors may lie on curved low-dimensional manifolds rather than in an unconstrained linear space.

This repository explores that idea through latent-variable models whose codes live on:

- a **sphere**,
- or a **torus**.

---

### 3.2 Variational Autoencoders

A Variational Autoencoder (VAE) learns:

- an **encoder** that maps input data to a latent distribution,
- a **decoder** that reconstructs the input from a sampled latent code.

In standard form, the objective is:

$$ \mathcal{L}(x) = \mathbb{E}_{q_\phi(z \mid x)} [\log p_\theta(x \mid z)] -\beta D_{\mathrm{KL}}\left(q_\phi(z \mid x)\,\|\,p(z)\right) $$

where:

- $q_\phi(z \mid x)$ is the encoder,
- $p_\theta(x \mid z)$ is the decoder,
- $p(z)$ is the prior distribution,
- $\beta$ controls the KL regularization strength.

In this repository, the latent prior is adapted to non-Euclidean settings rather than being restricted to a standard Gaussian in $\mathbb{R}^d$.

---

### 3.3 Spherical and Toroidal Latent Spaces

Two important latent geometries appear in the repository:

#### Spherical latent space

A spherical latent code lies on a sphere:

$$
z \in \mathbb{S}^{d-1}
$$

This is useful for directional or normalized representations, where only orientation matters.

#### Toroidal latent space

A toroidal latent code can be understood as a product of circles:

$$
\mathbb{T}^n = \underbrace{\mathbb{S}^1 \times \cdots \times \mathbb{S}^1}_{n \text{ times}}
$$

This is particularly relevant when latent variables encode **periodic structure**, such as angles or cyclic patterns.

The presence of files such as `von_mises_fisher.py` and geometry-specific VAE implementations suggests that the project studies probability distributions and latent sampling strategies adapted to these curved spaces.

---

### 3.4 Training Objective

The training configuration in `main.py` indicates a loss balancing several terms:

- reconstruction,
- KL regularization,
- latent-space regularization.

At a high level, this can be summarized as:

$$
\mathcal{L}
=
\gamma \, \mathcal{L}_{\mathrm{rec}}
+
\beta \, \mathcal{L}_{\mathrm{KL}}
+
\alpha \, \mathcal{L}_{\mathrm{latent}}
$$

where:

- $\mathcal{L}_{\mathrm{rec}}$ encourages faithful reconstruction,
- $\mathcal{L}_{\mathrm{KL}}$ regularizes the latent posterior,
- $\mathcal{L}_{\mathrm{latent}}$ encourages structure compatible with the chosen manifold,
- $\alpha$, $\beta$, and $\gamma$ control the tradeoff between these terms.

The configuration also includes parameters such as latent dimension, embedding dimension, encoder/decoder depth, dropout, and dataset choice, indicating a flexible experimental pipeline.

---

## 4. Code Walkthrough

### 4.1 `main.py`

`main.py` is the main orchestration script of the repository.

Its role is to:

1. load the dataset through the `dataloader` package,
2. instantiate the appropriate model depending on the dataset,
3. configure the optimizer and scheduler,
4. launch either training or inference.

From the current code structure:

- `SphericalVAE` is selected for datasets such as `S1_dataset` and `S2_dataset`,
- `ToroidalVAE` is selected for `T2_dataset`.

The script also exposes a configuration dictionary containing:

- latent dimension,
- embedding dimension,
- encoder and decoder width/depth,
- batch size,
- number of epochs,
- optimization parameters,
- save path,
- training vs inference mode.

This makes `main.py` the natural starting point for reproducing experiments.

---

### 4.2 `model/`

The `model/` directory contains the core learning code:

- `spherical_vae.py` — implementation of the spherical latent-space VAE,
- `torus_vae.py` — implementation of the toroidal latent-space VAE,
- `train.py` — training routines,
- `infer.py` — inference and evaluation utilities.

This is the central modeling component of the repository.

Conceptually, the folder is responsible for:

- defining the encoder/decoder architecture,
- enforcing or parameterizing latent geometry,
- training the latent-variable model,
- running inference from trained checkpoints.

---

### 4.3 `dataloader/`

The `dataloader/` package contains:

- `data.py` — dataset loading logic,
- `utils.py` — helper functions for preprocessing and data handling.

This package abstracts the data pipeline away from the model definitions, making it easier to swap datasets or preprocessing strategies without modifying the training code directly.

---

### 4.4 `CNN_experiment/`

The `CNN_experiment/` folder contains a more self-contained experimental setup.

Its contents include:

- `train_experiment.py` — training driver,
- `dataset.py` — local dataset utilities,
- `circular_VAE.py` — another geometry-aware latent model,
- `von_mises_fisher.py` — directional-distribution utilities,
- `ive.py` — mathematical helper functions,
- `errors.py` — error or metric helpers,
- `saves/` — stored experiment outputs.

This folder appears to explore a more focused or earlier experimental branch around CNNs and circular / manifold-aware latent modeling.

---

### 4.5 `Semantic Manifolds/`

This folder currently contains:

- `emotion_classification.ipynb`

This notebook suggests a more exploratory or application-oriented study of **semantic manifolds**, likely connecting latent geometry with a classification task in an emotion-analysis setting.

It complements the core model code by providing a notebook workflow for experimentation, visualization, and analysis.

---

## 5. Installation & Usage

### Clone the repository

```bash
git clone https://github.com/KHOUTAIBI/MedicalImageAnalysis.git
cd MedicalImageAnalysis
```

### Install common dependencies

A typical environment for this repository would include:

```bash
pip install torch torchvision numpy scipy matplotlib jupyter
```

For the CNN experiment folder, there is also a local `requirements.txt` file:

```bash
pip install -r CNN_experiment/requirements.txt
```

### Run the main script

```bash
python main.py
```

By default, the current `main.py` configuration is set up with explicit parameters for:

- dataset selection,
- latent size,
- model width/depth,
- optimization,
- inference vs training.

You can modify the configuration dictionary directly in `main.py` to switch experiments.

### Launch the notebook

For the semantic-manifolds notebook:

```bash
jupyter notebook "Semantic Manifolds/emotion_classification.ipynb"
```

---

## 6. Notes

- The repository currently has a minimal top-level README, so this document is intended to provide a clearer high-level overview.
- The codebase is best understood as a **research / coursework repository** rather than a polished package.
- The most distinctive aspect of the repository is the use of **non-Euclidean latent geometries** such as spherical and toroidal manifolds.
- This makes the project relevant not only to medical image analysis, but also more broadly to **representation learning**, **latent-variable modeling**, and **geometric deep learning**.
