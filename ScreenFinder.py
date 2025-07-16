import cv2
import numpy as np
from ImageHandler import ImageHandler

import time
import matplotlib.pyplot as plt
from typing import Tuple

class ScreenFinder:
    def __init__(self, edges: np.ndarray):
        self.processed_edges = edges
        h, w = edges.shape
        self.centre = (w // 2, h // 2)  # x, y

        self.hough_lines = None
        self.v_hough_lines = None
        self.h_hough_lines = None
        self.bounding_box = None

    def find_hough_lines(self, rho: float = 1.0, theta: float = np.pi/180, thresh: int = 100) -> np.ndarray:  # TODO currently just vertical and horizontal, but maybe want to do pairs of perpendiculars with ~5 degree separation
        """
        Finds suitable Hough lines - those close to vertical and horizontal. rho and theta are the search discretisation, thresh is the threshold
        """
        lines = cv2.HoughLines(self.processed_edges, rho, theta, thresh)

        if lines is None:
            raise ValueError("No Hough Lines could be found")

        # Also get near vertical and near horizontal lines
        angle_thresh = np.pi / 18  # 10 degrees
        thetas = lines[:, 0, 1]
        v_mask = (np.abs(thetas) < angle_thresh) | (np.abs(thetas - np.pi) < angle_thresh)  # Vertical lines are near 0 or 180
        h_mask = (np.abs(thetas - np.pi / 2) < angle_thresh)  # Horizontal lines are near 90

        self.v_hough_lines = lines[v_mask]
        self.h_hough_lines = lines[h_mask]
        self.hough_lines = lines

        return self.hough_lines
    
    def find_smallest_bounding_box(self, target_point: Tuple[int, int] | None = None):
        """
        Extracts the nearest Hough lines (vertical, horizontal) to left, top, right, and below the central point
        
        Parameters
        ----------
            target_point: point to find closest bounding box around
        """  # TODO see find hough lines todo about vertical and horizontal
        # TODO Biases towards centre - reasonable but risky in some cases -, perhaps worth training a model to identify the screen instead
        
        if target_point is None:
            target_point = self.centre
        px, py = target_point

        # Split image into halves i.e. top-bottom and left-right
        h_rhos, h_thetas = self.h_hough_lines[:, 0, 0], self.h_hough_lines[:, 0, 1]
        top_mask = (h_rhos * np.sin(h_thetas)) < py
        bottom_mask = ~top_mask
        top_lines, bottom_lines = self.h_hough_lines[top_mask], self.h_hough_lines[bottom_mask]

        v_rhos, v_thetas = self.v_hough_lines[:, 0, 0], self.v_hough_lines[:, 0, 1]
        left_mask = (v_rhos * np.cos(v_thetas)) < px
        right_mask = ~left_mask
        left_lines, right_lines = self.v_hough_lines[left_mask], self.v_hough_lines[right_mask]
    
        # Find lines nearest image centre
        h, w = self.processed_edges.shape[:2]
        top_line = self._find_nearest_hough_line(top_lines, 0, np.pi/2, target_point=target_point)
        bottom_line = self._find_nearest_hough_line(bottom_lines, h, np.pi/2, target_point=target_point)
        left_line = self._find_nearest_hough_line(left_lines, 0, 0, target_point=target_point)
        right_line = self._find_nearest_hough_line(right_lines, w, 0, target_point=target_point)

        self.hough_lines = np.array([top_line, bottom_line, left_line, right_line])
    
    def _find_nearest_hough_line(self, lines: np.ndarray, default_rho: float, default_theta: float, target_point: Tuple[int, int] | None = None) -> Tuple[float, float]:
        """
        Finds the nearest hough line defined using rho, thetas to the target point - target point defaults to centre. If lines are empty returns the defined default

        Parameters
        ----------
            lines: hough lines to be searched through
            default_rho: default rho value if no line found
            default_thetas: default theta value if no line found
            target_point: optional alternative target point to centre

        Returns
        -------
            line: tuple of (rho, theta)
        """
        if target_point is None:
            target_point = self.centre
        
        target_point = np.array(target_point)  # Convert to array for matrix operations

        if lines.size > 0:
            rhos, thetas = lines[:, 0, 0], lines[:, 0, 1]
            directions = np.vstack((np.cos(thetas), np.sin(thetas))).T
            distances = np.abs(directions @ target_point - rhos)
            nearest_line = lines[np.argmin(distances)]
        else:
            nearest_line = np.array([(default_rho, default_theta)])
        
        return nearest_line
    
    

    def show_hough_lines(self, target_img: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
        """
        Draws the hough lines found in self.lines on a copy of the target image

        Parameters
        ----------
            target_img: image to draw on
            color: color for the lines to be drawn in
            thickness: line thickness
        
        Returns
        -------
            The image with the Hough lines drawn on
        """
        if self.hough_lines is None:
            raise ValueError("No hough lines found to draw")
        
        rho, theta = self.hough_lines[:, 0, 0], self.hough_lines[:, 0, 1]
        # Convert polar to cartesian
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = (x0 + 1000 * (-b)).astype(int), (y0 + 1000 * (a)).astype(int)
        x2, y2 = (x0 - 1000 * (-b)).astype(int), (y0 - 1000 * (a)).astype(int)

        target_img_cp = target_img.copy()
        for idx in range(len(self.hough_lines)):
            cv2.line(target_img_cp, (x1[idx], y1[idx]), (x2[idx], y2[idx]), color=color, thickness=thickness)

        cx, cy = self.centre
        cv2.circle(target_img_cp, (cx, cy), radius = 50, color=(255, 255, 255), thickness = 10)
        
        return target_img_cp



if __name__ == "__main__":
    path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/March25Erg.jpg"
    img = cv2.imread(path)

    PREPROCESS_PARAMS = {  # Erg screen specific
    "resize": (0.25),
    "equalize": ("clahe", 2.0, (8, 8)),
    "blur": (31, 31),
    "edges": (25, 75),
    }

    img_handler = ImageHandler(img)

    # Process image as needed
    handler = img_handler.to_gray(target=img_handler.img)

    scale = PREPROCESS_PARAMS["resize"]
    handler = img_handler.resize((scale, scale), target=handler.gray)
    
    eq_mode, clip_limit, tile_size = PREPROCESS_PARAMS["equalize"]
    handler.equalize(eq_mode, clip_limit, tile_size, target=handler.resized)

    blur_size = PREPROCESS_PARAMS["blur"]
    handler.blur(blur_size, target=handler.equalized)

    thresh_low, thresh_high = PREPROCESS_PARAMS["edges"]
    handler.find_edges(thresh_low, thresh_high, target=handler.blurred)

    screen_finder = ScreenFinder(handler.edges)
    # img_handler.display("edges")
    screen_finder.find_hough_lines()
    screen_finder.find_smallest_bounding_box()


    disp_img = screen_finder.show_hough_lines(img_handler.resized)
    plt.imshow(disp_img, cmap='gray')
    plt.show()
