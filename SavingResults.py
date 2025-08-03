"""
SavingResults.py

Functions to create timestamped results folders and save binary prediction masks as PNG images.

Author: Margarita Cordova Rojas
Date: 2025-08-03
"""

# ========== Imports ==========
import os
import imageio
import numpy as np
from datetime import datetime

# ========== Functions ==========

def createResultsFolder():
    """
    Create a timestamped folder in the current working directory to save prediction results.

    Returns:
        pred_folder (str): Full path to the created (or existing) results folder.
    """

    # Get current date and time formatted as 'YYYY-MM-DD HH:MM:SS'
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get current working directory path
    base_dir = os.getcwd()

    # Construct a descriptive folder name including timestamp
    folder_name = f"MLP Predictions - {timestamp}"

    # Full path to the predictions folder
    pred_folder = os.path.join(base_dir, folder_name)

    # Create the folder if it doesn't already exist (no error if it does)
    os.makedirs(pred_folder, exist_ok=True)

    print(f"Saving predictions to: {pred_folder}")
    return pred_folder


def saveResults(predictions, filenames, save_dir):
    """
    Save each binary prediction mask as a black & white PNG file with the original filenames.

    Args:
        predictions (list of np.ndarray): List of predicted binary masks (arrays).
        filenames (list of str): Corresponding original image filenames.
        save_dir (str): Directory where PNG files will be saved.
    """

    # Iterate over each prediction and its filename
    for i, pred in enumerate(predictions):

        # Extract base filename without extension (e.g., 'image1' from 'image1.tif')
        base_name = os.path.splitext(filenames[i])[0]

        # Construct full path for the output PNG file
        filename = os.path.join(save_dir, f"{base_name}.png")

        # Convert boolean mask (True=1) to uint8 image with 0 and 255 pixel values for visibility
        binary_mask = (pred == 1).astype('uint8') * 255

        # Save the binary mask as a PNG image using imageio
        imageio.imwrite(filename, binary_mask)

        print(f"Saved prediction: {filename}")
