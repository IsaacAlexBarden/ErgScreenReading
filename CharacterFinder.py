import cv2
import skimage
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from typing import Tuple

from ImageHandler import ImageHandler
from ScreenFinder import ScreenFinder

class CharacterFinder:
    BINARIZE_PARAMS = {  # Erg screen specific
    "equalize": ("clahe", 2.0, (8, 8)),
    "blur": (7, 7),
    "binarize": (255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
    }
    def __init__(self, screen_image):
        self.screen_image = screen_image

        self.bin_image = None
        self.horizontal_lines = None

    def _pre_process(self):
        handler = ImageHandler(self.screen_image)
        
        mode, clipLimit, tileSize = CharacterFinder.BINARIZE_PARAMS["equalize"]
        handler.equalize(mode, clipLimit, tileSize, target=handler.img)

        blur_kernel = CharacterFinder.BINARIZE_PARAMS["blur"]
        handler.blur(blur_kernel, target=handler.equalized)

        max_val, adpt_method, thresh_type, block_size, C = CharacterFinder.BINARIZE_PARAMS["binarize"]
        self.bin_image = cv2.adaptiveThreshold(handler.blurred, maxValue=max_val, adaptiveMethod=adpt_method,
                                          thresholdType=thresh_type, blockSize=block_size, C=C)
        self.bin_image = 1 - self.bin_image // 255
        # Save the binary we are working on for review
        cv2.imwrite("/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/Working Binary.jpeg", self.bin_image * 255)

    def _clean_binary(self):
        self._remove_large_regions()
        self._remove_small_specks()

    
    def _remove_large_regions(self) -> None:
        """
        Helper function to remove large connceted regions from the binary image
        A region is considered large if it is very long, wide, or large area
        """
        # Reconnect disconnected bars
        kernel_size = (3, 3)  # TODO kernel size depends on image size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        closed = cv2.morphologyEx(self.bin_image, cv2.MORPH_CLOSE, kernel)

        h, w = self.bin_image.shape
        max_allowable_height = h // 4  # Magic number perhaps - set as option?
        max_allowable_width = w // 4  # TODO numbers as function params?
        max_allowable_area = self.bin_image.size // 256
        
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for label_num in range(n_labels):
            if stats[label_num, cv2.CC_STAT_WIDTH] > max_allowable_width:
                self.bin_image[labels == label_num] = 0
            
            if stats[label_num, cv2.CC_STAT_HEIGHT] > max_allowable_height:
                self.bin_image[labels == label_num] = 0
            
            if stats[label_num, cv2.CC_STAT_AREA] > max_allowable_area:
                self.bin_image[labels == label_num] = 0
        return
    
    def _remove_small_specks(self) -> None:
        """
        Helper function to remove small specks of noise from the binary image
        a speck is a region of small area that is far from any non-speck.
        We include distance checks to keep colons and similar present in text in the image
        as these are close to non-speck regions
        """
        # First open to remove the tiniest specks
        kernel_size = (3, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        opened = cv2.morphologyEx(self.bin_image, cv2.MORPH_OPEN, kernel)

        # Second remove regions too far from characters - characters are of a certain area
        h, w = self.bin_image.shape
        min_char_area_ratio = 50 / 300000  # Based on July08Erg TODO numbers here as function params?
        min_area_for_char = self.bin_image.size * min_char_area_ratio
        char_height_estimate = h * 0.02  # 2% of image height
        char_width_estimate = w * 0.02  # 2% of image width
        max_allowable_dist_squared = (2 * char_height_estimate)**2 + (2 * char_width_estimate) ** 2  # Within 2 heights / 2 widths
        
        _, labelled, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
        is_small = stats[:, cv2.CC_STAT_AREA] < min_area_for_char
        is_large = ~is_small

        small_indices = np.where(is_small)[0]
        large_indices = np.where(is_large)[0]
        small_centroids = centroids[small_indices]
        large_centroids = centroids[large_indices]

        # If nothing large is found we have no characters
        if len(large_centroids) == 0:
            raise ValueError("No Characters Found")
        
        x_dists = small_centroids[:, 0:1] - large_centroids[:, 0]  # Slice this way to broadcast
        y_dists = small_centroids[:, 1:2] - large_centroids[:, 1]
        euclid_dists_squared = x_dists**2 + y_dists**2

        should_remove_speck = np.all(euclid_dists_squared > max_allowable_dist_squared, axis=1)
        for label, remove in zip(small_indices, should_remove_speck):
            if remove:
                opened[labelled == label] = 0
        
        self.bin_image = opened
        return
    
    def _separate_by_horizontal_line(self):
        """
        Helper function to split the binary image into horizontal lines by text
        Joins together components whose centroids lie within some threshold of one another
        When we are no longer within the same line, we start a new line
        """
        _, labelled, stats, centroids = cv2.connectedComponentsWithStats(self.bin_image, connectivity=8)
        
        # Ignore the background
        stats = stats[1:]
        centroids = centroids[1:]
        centroid_order = np.argsort(centroids[:, 1]) if len(centroids) else []

        line_height_fraction = 25/550  # ~25 pixels high in a 550 pixel high image
        in_same_line_threshold = line_height_fraction * self.bin_image.shape[0]  # Distance from current line centroid to region centroid

        # Consider making the check the larger of the fraction or the median component size
        comp_heights = stats[cv2.CC_STAT_HEIGHT, :]
        median_height = np.median(comp_heights)
        mean_height = np.mean(comp_heights)
        print(median_height, mean_height)

        # Build lines up
        horizontal_lines = []
        for i in centroid_order:
            label_id = i + 1  # true label
            cy = float(centroids[i, 1])
            
            best_idx, best_gap = None, None
            # Find best line for component
            for j, line in enumerate(horizontal_lines):
                gap = abs(cy - line["center"])
                if gap <= mean_height and (best_gap is None or gap < best_gap):
                    best_idx, best_gap = j, gap
                
            
            if best_idx is None:  # Create a new line
                horizontal_lines.append({"components": [label_id], "centroids": [cy], "center": float(cy)})
            else:
                line = horizontal_lines[best_idx]
                line["components"].append(label_id)
                line["centroids"].append(cy)
                line["center"] = float(np.mean(line["centroids"]))

        # Convert lines to ndarrays
        horizontal_line_images = []
        for line in horizontal_lines:
            mask = np.isin(labelled, line["components"]).astype(np.uint8) * 255
            y_coords, x_coords = np.where(mask > 0)
            if y_coords.size and x_coords.size:
                cropped = mask[y_coords.min():y_coords.max(), x_coords.min():x_coords.max()]
                horizontal_line_images.append(cropped)
            
        self.horizontal_lines = horizontal_line_images
        return
        
    def _separate_by_character(self):
        pass

    def _binarize(self):
        # Convert the image to binary and clean
        self._pre_process()
        self._clean_binary()

        plt.imshow(self.bin_image)
        plt.show()

        # Process horizontal lines
        self._separate_by_horizontal_line()
        for line in self.horizontal_lines:
            plt.imshow(line)
            plt.show()
        
        
        # Separate each line into characters !!!
        images_of_lines = []
        for line in line_groups:
            components = line['components']
            boxes = [stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_AREA] for i in components]  # Get LEFT, TOP, WIDTH, HEIGHT
            boxes_np = np.array(boxes)
            x_vals = boxes_np[:, 0]
            y_vals = boxes_np[:, 1]
            w_vals = boxes_np[:, 2]
            h_vals = boxes_np[:, 3]

            x1 = min(x_vals)
            y1 = min(y_vals)
            x2 = max([x + w for x, w in zip(x_vals, w_vals)])
            y2 = max([y + h for y, h in zip(y_vals, h_vals)])

            image_line = bin_image[y1:y2, x1:x2]
            images_of_lines.append(image_line)
        
        for line in images_of_lines:
            image_line = cv2.resize(line, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)  # Scale up to make getting characters easier
            eroded_line = cv2.morphologyEx(image_line, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=2)

            contours, _ = cv2.findContours(eroded_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = [cv2.boundingRect(c) for c in contours]
            bboxes = sorted(bboxes, key=lambda x: x[0])  # Sort by x-coordinate of TL corner

            merged = []
            for curr_box in bboxes:
                if not merged:
                    merged.append(curr_box)
                    continue
                
                # Check if this box should join previous
                last_box = merged[-1]

                if self._should_merge(curr_box, last_box, horizontal_gap_threshold=5, max_vertical_centroid_gap=50, min_aspect_ratio=0.2, max_aspect_ratio=2.0):
                    curr_x, curr_y, curr_w, curr_h = curr_box
                    last_x, last_y, last_w, last_h = last_box

                    new_x = min(curr_x, last_x)
                    new_y = min(curr_y, last_y)
                    new_w = max(curr_x + curr_w, last_x + last_w) - new_x
                    new_h = max(curr_y + curr_h, last_y + last_h) - new_y
                    merged_box = (new_x, new_y, new_w, new_h)
                    merged[-1] = merged_box

                else:
                    merged.append(curr_box)
                
            for (x, y, w, h) in merged:
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                colour_img = cv2.cvtColor(image_line * 255, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(colour_img, top_left, bottom_right, color=(0, 255, 0), thickness=2)
                plt.imshow(colour_img)
                plt.show()

    def _should_merge(self, box_a, box_b, horizontal_gap_threshold: int, max_vertical_centroid_gap: int, min_aspect_ratio: float, max_aspect_ratio: float) -> bool:
        """
        Helper function to rejoin characters separated by binarisation
        Uses horizontal gap and aspect ratio to judge if bounding boxes belong together
        """
        return self._is_horizontal_gap_small(box_a, box_b, horizontal_gap_threshold) \
                and self._is_vertical_centre_gap_small(box_a, box_b, max_vertical_centroid_gap) \
                and self._aspect_ratio_allowed(box_a, box_b, min_aspect_ratio, max_aspect_ratio)

    def _is_horizontal_gap_small(self, box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int], max_gap: int = 5) -> bool:
        """
        Helper function for merging separated character blobs into one character - checks horizontal overlap

        Parameters
        ----------
            box_a : The current box, tuple of ints representing TL corner as x,y and width and height
            box_b : The previous box, tuple of ints representing TL corner x,y and width and height
            gap_threshold: Integer representing maximum allowable gap
        
        Returns
        -------
            True if the gap is less than or equal to the threshold
            False otherwise
        """
        # TODO convert a and b to curr and prev???
        x_a = box_a[cv2.CC_STAT_LEFT]  # TODO check if this or my own enum is better
        x_b, w_b = box_b[cv2.CC_STAT_LEFT], box_b[cv2.CC_STAT_WIDTH]

        # Check boxes in correct order
        if x_a < x_b:
            raise ValueError(f"Box A: {box_a}, begins further left than Box B: {box_b}. Please check the ordering")
        
        gap = x_a - (x_b + w_b)
        if gap <= max_gap:
            return True
        else:
            return False  # TODO more pythonic to not else?

    def _is_vertical_centre_gap_small(self, box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int], max_gap: int = 5) -> bool:
        y_a, height_a = box_a[cv2.CC_STAT_TOP], box_a[cv2.CC_STAT_HEIGHT]
        y_b, height_b = box_b[cv2.CC_STAT_TOP], box_b[cv2.CC_STAT_HEIGHT]

        centre_a = y_a + height_a / 2
        centre_b = y_b + height_b / 2

        if abs(centre_a - centre_b) <= max_gap:
            return True
        else:
            return False  # More pythonic to just return False?

    def _aspect_ratio_allowed(self, box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int], min_aspect_ratio: float = 0.2, max_aspect_ratio: float = 2.0) -> bool:
        """
        Helper function for merging separated character blobs into one character - checks aspect ratio (width/height)

        Parameters
        ----------
            box_a : The current box, tuple of ints representing TL corner as x,y and width and height
            box_b : The previous box, tuple of ints representing TL corner x,y and width and height
            min_aspect_ratio: Float representing the minimum allowable aspect ratio for a valid character
            max_aspect_ratio: Float representing the maximum allowable aspect ratio for a valid character
        
        Returns
        -------
            True if the aspect ratio is within the bounds
            False otherwise
        """
        x_a, y_a, width_a, height_a = box_a
        x_b, y_b, width_b, height_b = box_b

        if x_a < x_b:
            raise ValueError(f"Box A: {box_a}, begins further left than Box B: {box_b}. Please check the ordering")

        merged_width = max(x_a + width_a, x_b + width_b) - min(x_a, x_b)
        merged_height = max(y_a + height_a, y_b + height_b) - min(y_a, y_b)
        aspect_ratio = merged_width / merged_height

        if min_aspect_ratio <= aspect_ratio and aspect_ratio <= max_aspect_ratio:
            return True
        else:
            return False  # TODO more pythonic to avoid else


if __name__ == "__main__":
    import time
    t1 = time.perf_counter()
    path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/March25Erg.jpg"
    path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/April15Erg.jpg"
    #path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/July08Erg.jpg"
    img = cv2.imread(path)

    PREPROCESS_PARAMS = {  # Erg screen specific
    "resize": (0.25),
    "equalize": ("clahe", 2.0, (8, 8)),
    "blur": (7, 7),
    "edges": (50, 75),
    }

    img_handler = ImageHandler(img)

    # Process image as needed
    t2 = time.perf_counter()
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
    print(time.perf_counter() - t2)

    warp_handler = ImageHandler(warped)
    warp_handler.to_gray(target=warp_handler.img)
    warp_handler.resize(scale, target=warp_handler.gray)
    char_extract = CharacterFinder(warp_handler.resized)
    char_extract._binarize()


