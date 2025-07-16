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

        self.bounding_lines = None
        self.intersections = None
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
    
    def find_nearest_bounding_lines(self, target_point: Tuple[int, int] | None = None):
        """
        Extracts the nearest Hough lines (vertical, horizontal) to left, top, right, and below the target points
        
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

        self.bounding_lines = np.array([top_line, bottom_line, left_line, right_line])
    
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
    
    def generate_warp_matrix(self, show_box: bool = False, target_image: np.ndarray | None = None) -> np.ndarray:
        """
        Uses the four nearest to centre hough lines to generate the warp matrix for the image to the erg screen
        at the provided scale - optionally displays the bounding box

        Parameters
        ----------
            show_box: toggle for displaying the bounding box
        
        Returns
        -------
            M: the warping matrix
            targ_w: the target height
            targ_h: the target width
        """
        # Get intersections
        self.intersections = self._get_bounding_intersections()

        # Intersection points in TL, TR, BL, BR order
        sorted_pts = self.intersections[np.argsort(self.intersections[:, 1])]
        TL, TR = sorted_pts[:2][np.argsort(sorted_pts[:2, 0])]
        BL, BR = sorted_pts[2:][np.argsort(sorted_pts[2:, 0])]

        # Crop image
        source_pts = np.array([TL, TR, BL, BR], dtype="float32")
        targ_w = max(int(np.linalg.norm(TL - TR)), int(np.linalg.norm(BL - BR)))
        targ_h = max(int(np.linalg.norm(TL - BL)), int(np.linalg.norm(TR - BR)))

        dest_pts = np.array([[0, 0],
                             [targ_w - 1, 0],
                             [0, targ_h - 1],
                             [targ_w - 1, targ_h - 1]],
                             dtype="float32")

        M = cv2.getPerspectiveTransform(source_pts, dest_pts)

        if show_box:
            if target_image is None:
                raise ValueError("No target image to draw box on")
            rect = cv2.minAreaRect(np.array(self.intersections))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(target_image, [box], 0, (0, 255, 0), 2)
            plt.imshow(target_image)
            plt.show()

        return M, targ_w, targ_h

    def _get_bounding_intersections(self, atol=1e-10) -> np.ndarray:
        """Helper method to find the intersections of the bounding lines"""
        # Confirm bounding lines exist
        if self.bounding_lines is None:
            raise ValueError("No bounding lines to use")
        
        # Get intersections - lines are in top, bottom, left, right order
        intersections = []
        h_lines = self.bounding_lines[:2]
        v_lines = self.bounding_lines[2:]

        for v_rho, v_theta in v_lines[:, 0]:
            for h_rho, h_theta in h_lines[:, 0]:
                A = np.array([[np.cos(v_theta), np.sin(v_theta)],
                              [np.cos(h_theta), np.sin(h_theta)]])
                b = np.array([v_rho, h_rho])

                det = np.linalg.det(A)
                if np.isclose(det, 0, atol=atol):
                    raise ValueError("No interesection found for some lines, lines are near parallel")
                
                x, y = np.linalg.solve(A, b)
                intersections.append((x, y))
        
        # Check intersections in bounds
        h, w = self.processed_edges.shape[:2]
        tol = 10
        for x, y in intersections:
            if (x + tol) < 0 or (x - tol) > w or (y + tol) < 0 or (y - tol) > h:
                raise ValueError("Intersections outside image, cannot find suitable intersections")
        
        return np.array(intersections)

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
    import time
    path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/March25Erg.jpg"
    img = cv2.imread(path)

    PREPROCESS_PARAMS = {  # Erg screen specific
    "resize": (0.25),
    "equalize": ("clahe", 2.0, (8, 8)),
    "blur": (31, 31),
    "edges": (25, 75),
    }

    t1 = time.perf_counter()
    img_handler = ImageHandler(img)

    # Process image as needed
    t2 = time.perf_counter()
    handler = img_handler.to_gray(target=img_handler.img)

    t3 = time.perf_counter()
    scale = PREPROCESS_PARAMS["resize"]
    handler = img_handler.resize((scale, scale), target=handler.gray)
    
    t4 = time.perf_counter()
    eq_mode, clip_limit, tile_size = PREPROCESS_PARAMS["equalize"]
    handler.equalize(eq_mode, clip_limit, tile_size, target=handler.resized)

    t5 = time.perf_counter()
    blur_size = PREPROCESS_PARAMS["blur"]
    handler.blur(blur_size, target=handler.equalized)

    t6 = time.perf_counter()
    thresh_low, thresh_high = PREPROCESS_PARAMS["edges"]
    handler.find_edges(thresh_low, thresh_high, target=handler.blurred)

    t7 = time.perf_counter()
    screen_finder = ScreenFinder(handler.edges)
    # img_handler.display("edges")
    t8 = time.perf_counter()
    screen_finder.find_hough_lines()

    t9 = time.perf_counter()
    screen_finder.find_nearest_bounding_lines()

    t10 = time.perf_counter()
    M, width, height = screen_finder.generate_warp_matrix()

    warped = cv2.warpPerspective(handler.resized, M, (width, height))
    plt.imshow(warped)
    plt.show()

    t_end = time.perf_counter()
    print(f"Time for ImageHandler creation was {t_end - t1}")
    print(f"Time for gray conversion was {t_end - t2}")
    print(f"Time for resizing was {t_end - t3}")
    print(f"Time for equalizing was {t_end - t4}")
    print(f"Time for blurring was {t_end - t5}")
    print(f"Time for edge finding was {t_end - t6}")
    print(f"Time for ScreenFinder creation was {t_end - t7}")
    print(f"Time for Finding Hough Lines - Vertical and Horizontal was {t_end - t8}")
    print(f"Time for Finding Nearest Bounding Lines was {t_end - t9}")
    print(f"Time for Creating the Bounding Box was {t_end - t10}")

    # disp_img = screen_finder.show_hough_lines(img_handler.resized)
    # plt.imshow(disp_img, cmap='gray')
    # plt.show()
