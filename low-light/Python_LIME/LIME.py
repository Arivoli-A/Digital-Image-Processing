import cv2
import numpy as np
from . import LIME_functions as LIME

from os import listdir
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from bm3d import bm3d

def lime(
        image,
        weight_strategy=3,
        gamma=0.4,
        std_dev=0.04
    ):
    """
    Process a single image using the LIME enhancement pipeline.

    Parameters
    ----------
    image : str
        Image in ope.
    weight_strategy : int
        Weight strategy for illumination map update.
    gamma : float
        Gamma for correction.
    std_dev : float
        Standard deviation parameter for BM3D denoising.

    Returns
    -------
    denoised_image : np.ndarray
        Enhanced and denoised image in RGB (float32, range 0–1).
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    #Compute illumination map
    illumination_map = np.max(image_rgb, axis=-1)
    updated_ill_map = LIME.update_illumination_map(
        illumination_map, weight_strategy)

    corrected_ill_map = LIME.gamma_correction(
        np.abs(updated_ill_map), gamma)[..., np.newaxis]

    #Image enhancement
    new_image = image_rgb / corrected_ill_map
    new_image = np.clip(new_image, 0, 1).astype("float32")

    #BM3D denoising
    denoised_image = LIME.bm3d_yuv_denoising(
        new_image, corrected_ill_map, std_dev)


    return denoised_image

