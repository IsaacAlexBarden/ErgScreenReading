import cv2
import numpy as np
from ImageHandler import ImageHandler

import time
import matplotlib.pyplot as plt
from typing import Tuple, List

class ScreenFinder:
    def __init__(self, edges: np.ndarray):
        self.edges = edges

        self.erg_screen = None

    def _get_candidate_screen_locations(self, min_area: int = 10000, min_aspect: float = 0.5, max_aspect: float = 1.8) -> List:
        """Extracts the erg screen bounding box from the supplied edges as a x, y, w, h tuple"""
        closed_edges = cv2.morphologyEx(self.edges, cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))  # TODO remove this magic number
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidate_screens = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            bbox = cv2.boundingRect(cnt)
            if self._is_valid_bbox(bbox, min_aspect=min_aspect, max_aspect=max_aspect):
                candidate_screens.append((area, cnt))
        
        return candidate_screens

    
    def _is_valid_bbox(self, bbox: Tuple[int, int, int, int], min_aspect: float, max_aspect: float) -> bool:
        x, y, w, h = bbox
        aspect = w / float(h)  # Ensure aspect doesnt round
        cx, cy = x + w // 2, y + h // 2  # TODO set distance to centre allowable?
        margin_x = self.edges.shape[1] // 4 # TODO magic numbers!
        margin_y = self.edges.shape[0] // 4

        if min_aspect < aspect < max_aspect and abs(cx - self.edges.shape[1]//2) < margin_x and abs(cy - self.edges.shape[0]//2) < margin_y:  # TODO if more checks make separate functions
            return True
    
    def get_erg_screen_warp_parameters(self, scale) -> Tuple:
        """Extracts a tuple of M, width, height for the warp to the erg screen"""
        candidates = self._get_candidate_screen_locations(min_area=10000, min_aspect=0.5, max_aspect=1.8)

        if not candidates:
            raise ValueError("No valid screen contour found")
        
        candidates.sort(reverse=True)  # Sort by decreasing area
        screen_contour = candidates[0][1]

        # Extract coordinates of bounding region
        rect = cv2.minAreaRect(screen_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # To system standard
        box = (box / scale).astype(np.float32)

        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            sum_ = pts.sum(axis=1)  # TL will be smallest, BR largest
            rect[0] = pts[np.argmin(sum_)]  # TODO enum for indexing these?
            rect[2] = pts[np.argmax(sum_)]
            diff = np.diff(pts, axis=1)  # y - x
            rect[1] = pts[np.argmin(diff)]  # TR is largest x, smallest y
            rect[3] = pts[np.argmax(diff)]  # BL is smallest x, largest y
            return rect

        rect = order_points(box)
        (TL, TR, BR, BL) = rect

        # Extract warp matrix
        width = int(max(np.linalg.norm(BR - BL), np.linalg.norm(TR - TL)))
        height = int(max(np.linalg.norm(TR - BR), np.linalg.norm(TL - BL)))

        dst = np.array([[0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        return M, width, height

if __name__ == "__main__":
    import time
    #path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/March25Erg.jpg"
    #path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/April15Erg.jpg"
    path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/July08Erg.jpg"
    img = cv2.imread(path)

    PREPROCESS_PARAMS = {  # Erg screen specific
    "resize": (0.25),
    "equalize": ("clahe", 2.0, (8, 8)),
    "blur": (7, 7),
    "edges": (50, 75),
    }
    t1 = time.perf_counter()
    handler = ImageHandler(img)
    handler.to_gray(target=handler.img)
    scale = PREPROCESS_PARAMS["resize"]
    handler.resize(scale, target=handler.gray)
    eq_mode, lim, size = PREPROCESS_PARAMS["equalize"]
    handler.equalize(eq_mode, lim, size, target=handler.resized)
    blur_kernel = PREPROCESS_PARAMS["blur"]
    handler.blur(blur_kernel, target=handler.equalized)
    thresh_low, thresh_high = PREPROCESS_PARAMS["edges"]
    handler.find_edges(thresh_low, thresh_high, target=handler.blurred)

    screen_finder = ScreenFinder(handler.edges)
    M, width, height = screen_finder.get_erg_screen_warp_parameters(scale=scale)
    warped = cv2.warpPerspective(handler.img, M, (width, height))
    print(time.perf_counter() - t1)

    plt.imshow(warped)
    plt.show()