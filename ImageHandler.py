import cv2
import os
import numpy as np
import functools
import matplotlib.pyplot as plt

from win32api import GetSystemMetrics  # TODO change to be cross system

from typing import Tuple, Callable

# TODO
# Rewrite targetting so that strings are passed rather than images
"""
crop
rotate
flip
normalize
apply??
float_32?
^^ For augmentation and ML
"""

def _resolve_target(default_resolver: Callable):
    """
    Decorator for resolving the 'target' parameter in image processing methods. Helps assign target images in the processing steps

    Parameters
    ----------
        default_resolver: a function taking self that returns a default image if the target is None
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, target=None, **kwargs):
            if target is None:
                target = default_resolver(self)
                if target is None:
                    raise ValueError("No target provided and default target not found")
            
            return func(self, *args, target=target, **kwargs)
        return wrapper
    return decorator

class ImageHandler:
    def __init__(self, img):
        self.img = img

        self.gray = None
        self.resized = None
        self.blurred = None
        self.normalized = None
        self.equalized = None
        self.edges = None

        self.save_dir = None

    @_resolve_target(lambda self: self.img)
    def to_gray(self, *, target: np.ndarray):
        """Converts the target image to grayscale - defaults to self.img if no target specified"""
        if self.gray is not None:
            return self
        
        if len(target.shape) == 3:
            self.gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        
        elif len(target.shape) == 4:
            self.gray = cv2.cvtColor(target, cv2.COLOR_BGRA2GRAY)
        
        else:
            self.gray = target  # already grayscale
        
        self._save_image(self.gray, "gray")
        return self

    @_resolve_target(lambda self: self.gray if self.gray is not None else self.img)
    def resize(self, scale: float | None = None, *, target: np.ndarray) -> np.ndarray:
        """
        Resize and store target image - default is to screen size maintaining aspect ratio. Scale provided as (sx, sy), and is multiplied
        Defaults to self.gray, then self.img if no target specified
        """        
        self.resized = self._resize_image(target, scale)
        self._save_image(self.resized, "resized")

        return self

    @_resolve_target(lambda self: self.resized if self.resized is not None else self.gray)
    def blur(self, blur_size: Tuple[int, int], sigmaX: float = 0, sigmaY: float = 0, *, target: np.ndarray):
        """
        Blurs the target image
        Defaults to self.resized, then self.gray if no image supplied
        """
        self.blurred = cv2.GaussianBlur(target, blur_size, sigmaX=sigmaX, sigmaY=sigmaY)
        self._save_image(self.blurred, "blurred")

        return self
    
    @_resolve_target(lambda self: self.blurred if self.blurred is not None else self.gray)
    def normalize(self, *, target: np.ndarray):
        """
        Performs normalization to 0-255 range
        Defaults to self.blurred if no target is provided, gray otherwise
        """
        self.normalized = cv2.normalize(target, None, 0, 255, cv2.NORM_MINMAX)
        self._save_image(self.normalized, "normalized")
        
        return self

    @_resolve_target(lambda self: self.normalized if self.normalized is not None else self.gray)
    def equalize(self, mode='clahe', *args, target: np.ndarray):
        """
        Performs histogram equalization, in standard or clahe. Clahe requires a clipLimit and tileSize provided also
        Defaults to self.normalized if no target provided, then self.gray
        """
        if mode != "standard" and mode != "clahe":
            raise ValueError("Invalid equalization mode")
        
        if mode == "standard":
            self.equalized = cv2.equalizeHist(target)
        
        if mode == "clahe":
            if len(args) != 2:
                raise ValueError("Clahe needs clipLimit and tileSize provided in that order")
            clipLimit, tileSize = args

            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileSize)
            self.equalized = clahe.apply(target)
        
        self._save_image(self.equalized, "equalised")
        return self

    @_resolve_target(lambda self: self.equalized if self.equalized is not None else self.gray)
    def find_edges(self, thresh_low: int = 50, thresh_high: int = 100, *, target: np.ndarray):
        """
        Finds image edges using Canny edge detector
        Defaults to self.equalized, then self.gray if no target provided
        """
        self.edges = cv2.Canny(target, thresh_low, thresh_high)
        self._save_image(self.edges, "edges")

        return self

    def _save_image(self, image: np.ndarray, step_name: str) -> bool:
        """Internal method to save intermediates if save_dir is set"""
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"{step_name}.png")
            cv2.imwrite(save_path, image)
            return True
        return False
    
    def _resize_image(self, image: np.ndarray, scale: float | None = None) -> np.ndarray:
        """Internal method for temporary resizing"""
        # Get scaling factors
        h, w = self.img.shape[:2]
        if scale is None:
            scale = 0.25

        return cv2.resize(image, None, fx=scale, fy=scale)

    def display(self, step_name: str, resize: bool = True, scale: Tuple[float, float] | None = None) -> None:
        """
        Method for displaying image, can be resized to certain scale if requested
        
        Parameters
        ----------
            step_name: name of the step to be displayed 'resized', 'gray', 'equalized', 'blur', 'edges'
            resize: should the image be resized? True or False, defaults to true
            scale: scale for resizing, x, y. Defaults to resize to screen dimensions
        """
        step_img = getattr(self, step_name, None)
        if step_img is None:
            raise ValueError(f"No image stored for step {step_name}")
        
        if resize:
            disp_img = self._resize_image(step_img, scale)
        else:
            disp_img = step_img
        
        cv2.imshow(step_name, disp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/March25Erg.jpg"
    img = cv2.imread(img_path)
    handler = ImageHandler(img)
    handler.to_gray(target=handler.img)
    handler.resize(scale=(0.25, 0.25), target=handler.gray)
    handler.blur(blur_size=(35, 35), target=handler.resized)
    handler.normalize(target=handler.blurred)
    handler.equalize("clahe", 2.0, (8, 8), target=handler.normalized)
    thresh = cv2.adaptiveThreshold(handler.equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    plt.imshow(thresh)
    plt.show()