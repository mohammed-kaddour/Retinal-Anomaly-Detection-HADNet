import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from typing import List

def extract_haralick(gray_img: np.ndarray) -> List[float]:
    """
    Extracts 6 Haralick texture features from the GLCM (Gray-Level Co-occurrence Matrix).
    Features are averaged across 4 angles (0, 45, 90, 135) for rotation invariance.
    
    Args:
        gray_img: Grayscale input image.
        
    Returns:
        List containing [Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM].
    """
    # Calculate GLCM
    glcm = graycomatrix(gray_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    
    # Extract properties and compute mean across angles
    return [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean(),
        graycoprops(glcm, 'ASM').mean()
    ]

def extract_lbp(gray_img: np.ndarray, P: int = 8, R: int = 1) -> List[float]:
    """
    Extracts Local Binary Patterns (LBP) to capture micro-structural patterns.
    Uses the 'uniform' method for robust feature representation.
    
    Args:
        gray_img: Grayscale input image.
        P: Number of circularly symmetric neighbor set points.
        R: Radius of circle.
        
    Returns:
        Normalized histogram of LBP codes.
    """
    lbp = local_binary_pattern(gray_img, P, R, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist.tolist()
