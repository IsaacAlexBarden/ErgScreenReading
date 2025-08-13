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
    "equalize": ("clahe", 1.5, (16, 16)),
    "blur": (1, 1),
    "binarize": (255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 16)
    }
    def __init__(self, screen_image):
        self.screen_image = screen_image

        self.bin_image = None
        self.horizontal_lines = None

    def _pre_process(self):
        handler = ImageHandler(self.screen_image)

        blur_kernel = CharacterFinder.BINARIZE_PARAMS["blur"]
        handler.blur(blur_kernel, target=handler.img)

        mode, clipLimit, tileSize = CharacterFinder.BINARIZE_PARAMS["equalize"]
        handler.equalize(mode, clipLimit, tileSize, target=handler.blurred)

        max_val, adpt_method, thresh_type, block_size, C = CharacterFinder.BINARIZE_PARAMS["binarize"]
        self.bin_image = cv2.adaptiveThreshold(handler.equalized, maxValue=max_val, adaptiveMethod=adpt_method,
                                          thresholdType=thresh_type, blockSize=block_size, C=C)

        # Save the binary we are working on for review
        cv2.imwrite("/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/WorkingBinary.jpeg", self.bin_image)


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
        _, labelled, stats, _ = cv2.connectedComponentsWithStats(self.bin_image, connectivity=8)
        small_speck_size = 5
        is_small = stats[:, cv2.CC_STAT_AREA] < small_speck_size
        small_indices = np.where(is_small)[0]
        small_indices = small_indices[small_indices != 0]
        mask_small = np.isin(labelled, small_indices)
        self.bin_image[mask_small] = 0
        return
    
    def _separate_by_horizontal_line(self):
        """
        Helper function to split the binary image into horizontal lines by text
        Joins together components whose centroids lie within some threshold of one another
        When we are no longer within the same line, we start a new line

        lines are stored in self.horizontal_lines as tuples of (image, (x, y, w, h))
        """
        kernel_size = 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(kernel_size, kernel_size))
        dilated = cv2.morphologyEx(self.bin_image, cv2.MORPH_DILATE, kernel)

        _, labelled, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)
        
        # Ignore the background
        stats = stats[1:]
        centroids = centroids[1:]
        centroid_order = np.argsort(centroids[:, 1]) if len(centroids) else []

        component_heights = stats[:, cv2.CC_STAT_HEIGHT]
        q25, q75 = np.percentile(component_heights, [25, 75])
        mask = (component_heights >= q25) & (component_heights <= q75)
        iq_mean_height = component_heights[mask].mean()

        # Build each line
        horizontal_lines = []
        for i in centroid_order:
            label_id = i + 1  # true label
            cy = float(centroids[i, 1])
            
            best_idx, best_gap = None, None
            # Find best line for component
            for j, line in enumerate(horizontal_lines):
                gap = abs(cy - line["center"])
                if gap <= iq_mean_height and (best_gap is None or gap < best_gap):
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
                cropped_mask = mask[y_coords.min():y_coords.max(), x_coords.min():x_coords.max()]
                cropped_bin_image = self.bin_image[y_coords.min():y_coords.max(), x_coords.min():x_coords.max()]
                cropped = cropped_mask & cropped_bin_image  # To remove dilation, and ignore off line elements

                # Store image and bounding box for image position
                horizontal_line_images.append((cropped, (x_coords.min(), y_coords.min(), x_coords.max() - x_coords.min(), y_coords.max() - y_coords.min())))
            
        self.horizontal_lines = horizontal_line_images
        return
        
    def _separate_by_character(self):
        """
        Helper function to split each line into its individual character segments
        Join together disparate components that have been separated by binarising

        Characters stored with bounding box in original image location
        """
        for line, (line_x, line_y, line_w, line_h) in self.horizontal_lines:
            scale_x, scale_y = 4, 4
            scaled_up_line = cv2.resize(line, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
            
            kernel_size = (1, 10)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=kernel_size)
            eroded_line = cv2.morphologyEx(scaled_up_line, cv2.MORPH_ERODE, kernel)

            contours, _ = cv2.findContours(eroded_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = [cv2.boundingRect(c) for c in contours]
            bboxes = sorted(bboxes, key=lambda x: x[0])  # By x-coordinate of top left

            merged = []
            for curr_box in bboxes:
                if not merged:
                    merged.append(curr_box)
                    continue
                
                last_box = merged[-1]
                if self._should_merge(curr_box, last_box, h_gap_threshold=5):
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

            for (char_x, char_y, char_w, char_h) in merged:

                top_left = (line_x + char_x // 4, line_y + char_y // 4)  # Rescale bounding box with //  4
                bottom_right = (top_left[0] + char_w // 4, top_left[1] + char_h // 4)
                colour_img = cv2.cvtColor(self.bin_image, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(colour_img, top_left, bottom_right, color=(0, 255, 0), thickness=2)
                
                
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(colour_img)
                plt.show()

    def _should_merge(self, curr_box: Tuple[int, int, int, int],
                      last_box: Tuple[int, int, int, int],
                      h_gap_threshold: int) -> bool:
        
        h_gap_check = self._is_horizontal_gap_small(curr_box, last_box, h_gap_threshold)

        return h_gap_check
    
    def _is_horizontal_gap_small(self, box_a: Tuple[int, int, int, int],
                                 box_b: Tuple[int, int, int, int],
                                 h_gap_threshold: int) -> bool:
        """
        Helper function to check if two bounding boxes are close horizontally

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
        x_a = box_a[cv2.CC_STAT_LEFT]
        x_b, w_b = box_b[cv2.CC_STAT_LEFT], box_b[cv2.CC_STAT_WIDTH]

        if x_a < x_b:
            raise ValueError(f"Box A: {box_a}, begins further left than Box B: {box_b}. Please check ordering")

        gap = x_a - (x_b + w_b)
        if gap <= h_gap_threshold:
            return True
        return False

    def get_characters(self):
        # Convert the image to binary and clean
        t1 = time.perf_counter()
        self._pre_process()
        t2 = time.perf_counter()
        self._clean_binary()
        t3 = time.perf_counter()
        print(f"Time for pre processing {t2 - t1}")
        print(f"Time for cleaning {t3 - t2}")
        cv2.imwrite("CleanWorkingBinary.jpeg", self.bin_image)
        t4 = time.perf_counter()

        self._separate_by_horizontal_line()
        t5 = time.perf_counter()
        print(f"Time for horizontal line separation {t5 - t4}")

        for line, (_,_,_,_) in self.horizontal_lines:

            line_big = cv2.resize(line, None, fx=4, fy=4, interpolation=cv2.INTER_AREA)

            foreground = (line_big == 255).astype(np.uint8)
            vertical_projection = np.sum(foreground, axis=0).astype(np.float32)

            num_labels, labelled, stats, centroids = cv2.connectedComponentsWithStats(line_big, connectivity=8)
            widths = stats[:, cv2.CC_STAT_WIDTH]
            median_width = np.median(widths)

            heights = stats[:, cv2.CC_STAT_HEIGHT]
            median_height = np.median(heights)

            areas = stats[:, cv2.CC_STAT_AREA]
            median_area = np.median(areas)

            # Need to separate characters
            # Naive idea just use bounding box of each connected component region
            # Fails when characters separated vertically e.g. i or j, and when characters have become separated by binarisation
            # Combined i, j, : etc into one box by merging vertically i.e. if there is no/small horizontal gap and 
            merged_boxes = []
            for i in range(1, num_labels):
                bbox = cv2.boundingRect((labelled == i).astype(np.uint8))
                merged_boxes.append(bbox)
            
            merged_boxes.sort(key=lambda x: x[0])  # by x
            print(merged_boxes)

            # Merging vertically
            vertically_merged_boxes = []
            for curr_box in merged_boxes:
                if not vertically_merged_boxes:
                    vertically_merged_boxes.append(curr_box)
                    continue

                last_box = vertically_merged_boxes[-1]
                x1, y1, w1, h1 = curr_box
                x2, y2, w2, h2 = last_box
                
                if self._is_horizontal_gap_small(curr_box, last_box, h_gap_threshold=2):
                    new_x = min(x1, x2)
                    new_y = min(y1, y2)
                    new_w = max(x1 + w1, x2 + w2) - new_x
                    new_h = max(y1 + h1, y2 + h2) - new_y
                    box = (new_x, new_y, new_w, new_h)
                    vertically_merged_boxes[-1] = box
                else:
                    vertically_merged_boxes.append(curr_box)
            
            print(vertically_merged_boxes)


            
            for box in vertically_merged_boxes:
                color_line = cv2.cvtColor(line_big, cv2.COLOR_GRAY2BGR)
                x, y, w, h = box
                top_left = (x, y)
                bottom_right = (x + w, y + h)
                cv2.rectangle(color_line, top_left, bottom_right, color = (0, 255, 0), thickness=2)
                plt.imshow(color_line)
                plt.show()

            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].imshow(line_big)
            ax[1].plot(np.arange(0, len(vertical_projection), 1), vertical_projection)
            plt.show()


if __name__ == "__main__":
    import time
    t1 = time.perf_counter()
    path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/April15Erg.jpg"
    path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/July08Erg.jpg"
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
    char_extract.get_characters()



