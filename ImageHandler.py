import cv2
import os
import numpy as np
from win32api import GetSystemMetrics

from typing import Tuple

# TODO
"""
crop
rotate
flip
normalize
apply??
float_32?
^^ For augmentation and ML
"""

class ImageHandler:
    def __init__(self, img):
        self.img = img

        self.gray = None
        self.equalized = None
        self.blurred = None
        self.edges = None

        self.save_dir = None
    
    def to_gray(self):
        """Converts the image to grayscale"""
        if self.gray is not None:
            return self
        
        if len(self.img.shape) == 3:
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        elif len(self.img.shape) == 4:
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGRA2GRAY)
        
        else:
            self.gray = self.img  # already grayscale
        
        self._save_image(self.gray, "gray")
        return self


    def equalize(self, mode='clahe', *args):
        """Performs histogram equalization, in standard or clahe. Clahe requires a clipLimit and tileSize provided also"""
        if self.gray is None:  # Ensure we have a gray image
            raise ValueError("No gray image to equalize")

        if mode != "standard" and mode != "clahe":
            raise ValueError("Invalid equalization mode")
        
        if mode == "standard":
            self.equalized = cv2.equalizeHist(self.gray)
        
        if mode == "clahe":
            if len(args) != 2:
                raise ValueError("Clahe needs clipLimit and tileSize provided")
            clipLimit, tileSize = args

            clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileSize)
            self.equalized = clahe.apply(self.gray)
        
        self._save_image(self.equalized, "equalised")
        return self

    def blur(self, blur_size: Tuple[int, int], sigmaX: float = 0, sigmaY: float = 0):
        """Blurs the image provided, applys to equalize if available, gray otherwise"""
        target = self.equalized if self.equalized is not None else self.gray
        if target is None:
            raise ValueError("Must have gray or equalized to blur")
        
        self.blurred = cv2.GaussianBlur(target, blur_size, sigmaX=sigmaX, sigmaY=sigmaY)
        self._save_image(self.blurred, "blurred")

        return self

    def find_edges(self, thresh_low: int = 50, thresh_high: int = 100):
        """Finds image edges using Canny edge detector - operates on self.blurred if available, else self.gray"""
        target = self.blurred if self.blurred is not None else self.gray
        if target is None:
            raise ValueError("Must have gray or blurred image to find edges")
        
        self.edges = cv2.Canny(target, thresh_low, thresh_high)
        self._save_image(self.edges, "edges")

        return self

    def _save_image(self, image: np.ndarray, step_name: str):
        """Internal method to save intermediates if save_dir is set"""
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"{step_name}.png")
            cv2.imwrite(save_path, image)

    def _resize_image(self, image: np.ndarray, scale: Tuple[float, float] | None = None):
        """Internal method to resize image - default is to screen size maintaining aspect ratio. Scale provided as (sx, sy)"""
        h, w = self.img.shape[:2]
        if scale is None:
            screen_h, screen_w = GetSystemMetrics(1), GetSystemMetrics(0)
            scale_x, scale_y = screen_w / w, screen_h / h
            scale_factor = min(scale_x, scale_y)
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        
        else:
            new_w, new_h = int(w * scale[0]), int(h * scale[1])
        
        return cv2.resize(image, (new_w, new_h))

    def display_image(self, step_name: str, resize: bool = True, scale: Tuple[float, float] | None = None):
        """Method for displaying image, can be resized to certain scale if requested"""
        step_image = getattr(self, step_name, None)
        if step_image is None:
            raise ValueError(f"No image stored for step {step_name}")
        
        disp_img = self._resize_image(step_image, scale) if resize else step_image
        cv2.imshow(step_name, disp_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


            
            

if __name__ == "__main__":
    img_path = "sample_image.jpg"
    img = cv2.imread(img_path)

    handler = ImageHandler(img)
