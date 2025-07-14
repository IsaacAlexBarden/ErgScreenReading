import cv2
from ImageHandler import ImageHandler

from typing import Tuple

class ScreenFinder:
    PREPROCESS_PARAMS = {  # Erg screen specific
        "equalize": ("clahe", 2.0, (8, 8)),
        "blur": (31, 31),
        "edges": (25, 75),
    }


    def __init__(self, img):
        self.img_handler = ImageHandler(img)  # May make this an image handler class later
        self.processed_edges = self._preprocess()

        self.where
    
    def _preprocess(self):
        """Calls the necessary preprocessing for finding erg screens from the ImageHandler class"""
        handler = self.img_handler.to_gray()
        
        eq_mode, clip_limit, tile_size = self.PREPROCESS_PARAMS["equalize"]
        handler.equalize(eq_mode, clip_limit, tile_size)

        blur_size = self.PREPROCESS_PARAMS["blur"]
        handler.blur(blur_size)

        thresh_low, thresh_high = self.PREPROCESS_PARAMS["edges"]
        handler.find_edges(thresh_low, thresh_high)

        return handler.edges
        