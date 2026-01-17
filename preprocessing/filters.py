import cv2
import numpy as np

def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Removes black borders (background) from fundus images to focus on the 
    retinal area.
    
    Args:
        img: Input retinal image (BGR or Grayscale).
        tol: Tolerance threshold for black pixel intensity.
    
    Returns:
        Cropped image focusing on the informative area.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0): 
            return img 
        img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)

def hybrid_preprocess(image: np.ndarray, img_size: int = 256) -> np.ndarray:
    """
    Applies a hybrid enhancement pipeline:
    1. Ben Graham's method: Gaussian Blur subtraction to highlight lesions.
    2. CLAHE: Contrast Limited Adaptive Histogram Equalization for texture reinforcement.
    
    Args:
        image: Original retinal image.
        img_size: Target size for resizing.
        
    Returns:
        Enhanced RGB image.
    """
    # Standard resize
    img = cv2.resize(image, (img_size, img_size))
    
    # Ben Graham's Method (Highlights hemorrhages and exudates)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
    
    # Texture Enhancement using CLAHE in LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
