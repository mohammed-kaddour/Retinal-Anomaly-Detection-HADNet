import cv2
import numpy as np

def crop_image_from_gray(img, tol=7):
    """Removes black borders to focus on retinal texture."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if (img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0): return img 
        img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)

def hybrid_preprocess(image, img_size=256):
    """
    Applies Ben Graham's method and CLAHE to highlight lesions and textures.
    """
    img = cv2.resize(image, (img_size, img_size))
    
    # Ben Graham's Method (Gaussian Blur subtraction)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), 10), -4, 128)
    
    # CLAHE Enhancement
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
