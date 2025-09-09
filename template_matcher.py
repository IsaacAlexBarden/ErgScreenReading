import numpy as np
from typing import Dict
from enum import Enum, auto
import cv2

from extract_templates import TemplateType, template_preprocess

def flatfield(gray: np.ndarray, sigma: float = 15, eps: float = 1e-3) -> np.ndarray:
    """
    Correct the grayscale for variations in illumination e.g. weird lighting effects.

    Parameters
    ----------
        sigma : float
            blurring to use to get illumination
        eps : float
            small denominator addition to avoid division by zero in normalisation
    
    Returns
    -------
        norm : np.ndarray
            flattened version of the original grayscale i.e. brightness normalised
    
    """
    g = gray.astype(np.float32)
    illumination = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma)

    norm = (g / (illumination + eps)) * np.mean(illumination)
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return norm

def best_template_match(res: np.ndarray, template: np.ndarray, method: int, roi_offset: Tuple[int, int] = (0, 0)) -> Tuple[Tuple[int, int, int, int], int]:
    """
    roi_offset : if the region of interest is offset from the image top left the offset goes here

    Returns
    -------
        a tuple of bounding box and fit score
    """
    # Extract minima and maxima from res
    h, w = template.shape
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if method in {cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED}:
        top_left = min_loc
        score = 1.0 - float(min_val) if method == cv2.TM_SQDIFF_NORMED else -float(min_val)
    else:
        top_left = max_loc
        score = float(max_val)

    x, y = top_left[0], top_left[1]
    x0, y0 = x + roi_offset[0], y + roi_offset[1]

    bbox = (x0, y0, w, h)

    return bbox, score

class TemplateMatcher:
    def __init__(self, img: np.ndarray, templates: Dict[TemplateType, np.ndarray]):
        pass