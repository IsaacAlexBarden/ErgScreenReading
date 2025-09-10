"""
Script to extract regions of interest from a sample image (July08Erg). These are View Detail, meter, watt, time, /500m, s/m, 'heart'
Magic numbers are intentional, they all relate to the specified image_path, if this is changed all numbers must be adjusted
"""
import cv2
import numpy as np
from pathlib import Path
from enum import Enum, auto

TEMPLATE_BOUNDING_BOX = (417, 832, 2263, 2237)

class TemplateType(Enum):
    VIEW_DETAIL = auto()
    TIME = auto()
    METER = auto()
    PER_500_M = auto()
    S_PER_M = auto()
    HEART_RATE = auto()
    # WATT = auto() TODO find suitable template images
    # CAL = auto()

def template_preprocess(template: np.ndarray) -> np.ndarray:
    """Process the template, separate function for repeatability within template matcher."""
    return cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX)

if __name__ == "__main__":
    template_image_path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/July08Erg.jpg"
    img = cv2.imread(template_image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # July08Erg Specific slices for these ROIs
    # Template name : (y-slice, x-slice)
    roi_bounds = {
        TemplateType.VIEW_DETAIL: (slice(950, 1065), slice(562, 1212)),
        TemplateType.TIME: (slice(1461, 1574), slice(862, 1110)),
        TemplateType.METER: (slice(1485, 1574), slice(1224, 1544)),
        TemplateType.PER_500_M: (slice(1453, 1587), slice(1615, 1958)),
        TemplateType.S_PER_M: (slice(1388, 1577), slice(2022, 2208)),
        TemplateType.HEART_RATE: (slice(1468, 1566), slice(2287, 2391)),
        }

    save_dir = Path("Templates")
    for template_type, bounds in roi_bounds.items():
        # Extract and preprocess
        roi_img = img[bounds[0], bounds[1]]
        roi_img = template_preprocess(roi_img)

        # Save template
        save_path = save_dir / (template_type.name + ".png")
        cv2.imwrite(save_path, roi_img)
