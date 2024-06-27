# Discriminative Cross-Spectrum FaceRecSys

## Introduction

Cross-spectrum face recognition system using fused loss function for discriminative feature learning and ranking-based subspace hashing

 This project is the implementation of the paper <https://ieeexplore.ieee.org/document/9411963>, by *wang. H, et al (2021)*, which focuses on developing a robust face recognition system that operates across different imaging domains, specifically visible and thermal images. Utilizing state-of-the-art deep learning techniques, the system employs discriminative feature learning and ranking-based subspace hashing to achieve high accuracy in cross-modal face recognition tasks.

## Key Techniques

* **Discriminative Feature Learning:** Enhances the ability of the model to distinguish between different faces by learning discriminative features.
* **Ranking-Based Subspace Hashing:** Projects feature vectors into a lower-dimensional subspace, facilitating efficient and accurate face matching.
* **VGGFace Pretrained Models:** Used for initial feature extraction and embeddings to leverage pre-trained deep learning models.
* **Streamlit Application:** Provides an interactive interface for users to upload and process images, and visualize recognition results.
