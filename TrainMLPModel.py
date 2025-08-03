"""
TrainMLPModel.py

Functions for feature extraction from EcoFAB images and training an MLP classifier for root segmentation.

Author: Margarita Cordova Rojas
Date: 2025-08-03
"""

# ========== Imports ==========
# Note: Assume main script handles imports for skimage, sklearn, and others
import numpy as np
from functools import partial
from sklearn.neural_network import MLPClassifier
from skimage import feature, future

# ========== Functions ==========

def featureExtraction(sigmaMin, sigmaMax, rawImage):
    """
    Extract multiscale features from a raw EcoFAB image using skimage.

    Parameters:
        sigmaMin (float): Minimum sigma for multiscale features.
        sigmaMax (float): Maximum sigma for multiscale features.
        rawImage (ndarray): Raw RGB or grayscale input image.

    Returns:
        features (ndarray): Extracted feature array suitable for segmentation.
    """
    print("🌱 Extracting Multiscale Features from EcoFAB Training Images...")

    features_func = partial(
        feature.multiscale_basic_features,
        intensity=True,
        edges=True,
        texture=True,
        sigma_min=sigmaMin,
        sigma_max=sigmaMax,
        channel_axis=-1  # Channel axis for RGB images
    )
    
    features = features_func(rawImage)
    
    print("✅ Task Completed Successfully!")
    return features

def trainMLPModel(gt, features, mlp_parameters):
    """
    Train an MLP classifier using extracted features and ground truth masks.

    Parameters:
        features (ndarray): Feature array extracted from raw images.
        gt (ndarray): Ground truth mask image (background=0, root=255).

    Returns:
        trained_clf (MLPClassifier): Trained MLP segmentation classifier.
    """
    # Adjust ground truth mask labels to sklearn-friendly format (e.g., 1 and 2)
    gtMask = gt # (background = 0, root = 255)
    trainingLabels = gtMask.astype(np.uint8) + 2 # (0 -> 1, 255 -> 2)

    print("Training initialized...")
    
    # Train using skimage.future's fit_segmenter utility
    trained_clf = future.fit_segmenter(trainingLabels, features, mlp_parameters)
    
    print("Training complete.")
    
    return trained_clf
