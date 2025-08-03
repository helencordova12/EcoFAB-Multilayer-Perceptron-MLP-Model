"""
ImageUtils.py

This python file loads tif images and performs other helpful tasks for the images.

Author: Margarita Cordova Rojas
Date: 2025-08-03
"""

# ========== Imports ==========
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from functools import partial

# ========== Functions ==========

def loadImages(folderPath):
    """
    Loads all .tif images from a specified folder.

    Parameters:
        folderPath (str): Path to folder containing .tif images.

    Returns:
        images (list of ndarray): Loaded image arrays.
        filenames (list of str): Corresponding sorted filenames.
    """

    # Declare a list to store the loaded images
    images = []

    # Sort all filenames alphabetically
    filenames = sorted([f for f in os.listdir(folderPath) if f.lower().endswith('.tif') ]) 

    # Read each image and append it to the images list
    for filename in filenames:
        imgPath = os.path.join(folderPath, filename)
        print(f"Loading: {filename}")  # 🖨️ Print file name
        img = plt.imread(imgPath)
        images.append(img)

    print(f"\nTotal images loaded: {len(images)}") 
    
    # Returns list of tif images and corresponding filenames
    return images, filenames

def sliceEcoFABs(predictions, totalRoots):

    """
    Slice a vertically stacked prediction image into individual predictions.

    Parameters:
        predictions (ndarray): Stacked prediction image.
        totalRoots (int): Number of individual images expected.

    Returns:
        slicedImages (list of bool arrays): Binary root masks.
    """
    
    # Declare list to store individual predictions
    slicedImages = []  
    fullHeight = predictions.shape[0]

    # Retrieve height of each individual image
    singleHeight = fullHeight // totalRoots

    for i in range(totalRoots):
        top = i * singleHeight
        bottom = (i + 1) * singleHeight
        
        # Slice out a prediction
        singlePred = predictions[top:bottom, :]

        # Append individual prediction to list
        slicedImages.append(singlePred == 1)   

    return slicedImages

def displayPredictionOverRawEcoFab(pred, raw, fig_size):
    plt.figure(figsize=fig_size)  # Increase size (adjust numbers as needed)
    plt.imshow(raw)
    plt.imshow(pred, cmap='Reds', alpha=0.5)
    plt.axis('off')
    plt.title("Result Over Raw Image")
    plt.show()
    
