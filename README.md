# EcoFAB-Multilayer-Perceptron-MLP-Model
Automated Plant Root Segmentation: a Comparative Analysis between MLP Model and RhizoNet

# Overview

Plant root segmentation is fundamental to plant research, providing detailed insights into root structure and development. Traditional deep learning models like RhizoNET offer high accuracy but require significant technical expertise and high-performance GPU resources for training and deployment.

This project presents a lightweight alternative using a Multi-Layer Perceptron (MLP) that simplifies the model-building process and enables efficient training on personal laptops.

# Key Features

✅ Lightweight Model: Utilizes a simple MLP architecture for semantic segmentation.

🖥️ CPU-Trainable: Trains in ~16 minutes on a standard MacBook Air (no GPU needed).

📊 Small Dataset Requirement: Trained on only 10 annotated EcoFAB bottom images (~84% smaller than RhizoNET’s 61-image training set).

⚖️ Comparable Performance: Delivers segmentation performance similar to RhizoNET despite the simplified architecture.

🔄 Adaptable Pipeline: Easily extended to other root image datasets or simple segmentation tasks.

# Dataset

Total Images: 32 EcoFAB Root Scans

- Training Set: 10 images

- Testing Set: 22 images

These images were obtained from EcoFAB bottom-view scans and annotated for semantic segmentation. The dataset was specifically chosen to evaluate performance under limited data conditions.



Evaluation: Compared against results from a pretrained RhizoNET model.
