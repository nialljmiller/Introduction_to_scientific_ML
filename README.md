# Introduction to Scientific Machine Learning

This repository accompanies the talk **"Introduction to Scientific ML"**  
by **Dr. Niall Miller**  
nmille39@uwyo.edu / niall.j.miller@gmail.com  
https://nialljmiller.com

---

## Overview

This repo is a primer on using machine learning for actual scientific work — not just industry hype or kitten detection. It’s focused on demystifying the ML buzzwords and connecting them to real-world scientific workflows.

### Covered Topics

- What is Machine Learning (ML)?
- What is AI?
- What are Neural Networks?
- Why does any of this matter in science?
- How is "learning" just glorified fitting?
- Linear regression, MLPs, RNNs, GPs, and Transformers

---

## What is Machine Learning?

Machine learning is using maths to approximate learning. It's about finding the right parameters in a framework to produce a desired result — that’s it.

**Examples:**

- `scipy.optimize.curve_fit` – used in 95% of arXiv since 2010
- `y = mx + c` – still a solid model
- Least squares – baby’s first ML
- MCMC – probabilistic sampling machine
- Gaussian Processes – covariance learning machines
- RNNs – time-series prediction
- Transformers – autoregressive prediction monsters
- DDPMs – image generation

---

## Teaching a Computer

The idea:  
You want to find a function F(x) such that F(x) ≈ y.

Steps:

1. Guess F(x)
2. Compare prediction to true y (error = y - F(x))
3. Update F(x) using that error
4. Repeat until F(x) isn’t total garbage

That's it. Welcome to gradient descent.

---

## Linear Regression Example

Classic:

    Y = mX + c

We can code this and teach the machine to find optimal `m` and `c`.  
Yes, it’s ML. No, it’s not sexy. Yes, it works.

---

## How Do We Inform the Model?

From the loss (E), we compute:

    dE/dm
    dE/dc

Then update:

    m_new = m_old - L * dE/dm
    c_new = c_old - L * dE/dc

Where L is the learning rate.

This is the whole update mechanism behind most ML.

---

## Why This is Easy (and Why That’s a Problem)

Linear models work great — when the relationship is linear.

Real data is noisy, non-linear, and annoying.  
Example: Fitting star classifications with a linear function? Good luck.

---

## Gaussian Processes (GP)

Gaussian Processes:

- Define distributions over functions
- Don’t assume a fixed parametric form
- Fit to the *covariance* of the data

GPs are flexible and give you uncertainty estimates — no extra work needed.

They’re expensive, but nice.

---

## Deep Learning vs Classical

| Classical                     | Deep Learning                    |
|------------------------------|----------------------------------|
| Requires known equations     | Learns from scratch              |
| Easy to interpret            | Total black box                  |
| Light compute requirements   | Scales to GPU farms              |
| Needs hand-crafted features | Learns features automatically    |

Deep learning is just brute-forced curve fitting without a known function.

---

## Recurrent Neural Networks (RNN)

Add memory to a Neural Net and you get an RNN.

These are great for:

- Time series
- Sequences
- Things where past matters

Key types:

- RNN (vanilla)
- LSTM (long-term memory)
- GRU (gated update tricks)

---

## RNNs in Practice

You show X, it guesses Y.  
You tell it how wrong it was.  
It learns a bit.  
You repeat.

After N epochs: it kinda works.

Training time and performance depend heavily on:

- Number of input features
- Size of the model
- Length of training set

---

## Minimum Specs (RNN, MLP, etc.)

**Minimum:**

- CPU: 4-core
- RAM: 16 GB
- GPU: NVIDIA CUDA-compatible with 8GB VRAM

**Recommended:**

- CPU: 8-core (Ryzen 7 or i9)
- RAM: 64 GB
- GPU: Tesla V100 or similar (32+ GB VRAM)

---

## Decision Trees

Very different to deep nets. They:

- Split data based on decision rules
- Are interpretable
- Work great with structured data

Can also be used for classification and regression.  
Often paired with ensembles (e.g. Random Forests, XGBoost).

---

## Self-Supervised Learning

- No labels? No problem.
- Train models to understand structure and generate internal representations.
- Works well for scientific classification when labeled data is rare or ambiguous.

Examples: variable star classification, anomaly detection, etc.

---

## Generative AI in Science

Fake data, real insight:

- Synthetic light curves
- Interpolation and extrapolation
- Model testing and robustness analysis

Includes models like:

- DDPM (Diffusion models)
- GANs
- VAEs

---

## Transformers

Imagine an RNN… but sideways.

Good at:

- Sequence modeling (text, time series, etc.)
- Scaling
- Parallelism

Bad at:

- Interpretability
- Small-data efficiency

You’ll see them in GPT, BERT, T5, and basically every 2020s paper.

---

## In Summary

ML is just a set of tools for parameter fitting, classification, and prediction.

From `curve_fit()` to GPT, it's all just flavors of the same idea:

> Use errors to update models.

Scientific ML means choosing the right tools for your data and goals — not just slapping a neural net on everything.

---
