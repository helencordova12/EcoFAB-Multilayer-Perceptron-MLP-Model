# EcoFAB Root Segmentation Model (MLP Approach)
Automated Plant Root Segmentation: a Comparative Analysis between MLP Model and RhizoNet

# Overview

Plant root segmentation is a critical task in plant biology, offering insights into root morphology, growth patterns, and interactions with the environment. While advanced deep learning models like RhizoNET achieve high segmentation accuracy, they oftentimes require technical expertise and high-performance GPU hardware.

This project introduces a lightweight, accessible alternative using a Multi-Layer Perceptron (MLP) for EcoFAB root segmentation. The model is designed to be trained and deployed on standard hardware (e.g., MacBook Air), dramatically lowering the barrier to entry for researchers and students.

# Key Features

✅ Lightweight Model: Utilizes a simple MLP architecture for semantic segmentation.

🖥️ CPU-Trainable: Trains in ~16 minutes on a standard MacBook Air (no GPU needed).

📊 Ideal for Small Datasets: Trained on only 10 annotated EcoFAB bottom images (~84% smaller than RhizoNET’s 61-image training set).

⚖️ Comparable Performance: Delivers segmentation performance similar to RhizoNET despite the simplified architecture.

🔄 Adaptable Pipeline: Easily extended to other root image datasets or simple segmentation tasks.

# Dataset

Total Images: 32 EcoFAB Root Scans

- Training Set: 10 images

- Testing Set: 22 images

These images were obtained from EcoFAB bottom-view scans and annotated for semantic segmentation. The dataset was specifically chosen to evaluate performance under limited data conditions.



Evaluation: Compared against results from a pretrained RhizoNET model.
