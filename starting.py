import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load image
path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/March25Erg.jpg"
img = cv2.imread(path)
img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))

# Single channel and blur
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ksize = 31
blur = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=0)

# Histogram equalisation
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalised = clahe.apply(gray)
blur = clahe.apply(blur)

# Extract edges
edges = cv2.Canny(blur, 25, 75)
plt.imshow(edges)
plt.show()


# #--- Contours instead??? ---
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
#     #if len(approx) <= 4 and cv2.isContourConvex(approx):  # potential rectangle
#     area = cv2.contourArea(approx)
#     if area > 100:
#         cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
# cv2.imshow("Contours", img)
# cv2.waitKey()


lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
# for line in lines:
#     rho, theta = line[0]
#     a, b = np.cos(theta), np.sin(theta)
#     x0, y0 = a * rho, b * rho
#     x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
#     x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
    
#     if (rho - cx) < centre_band or (rho - cy) < centre_band:
#         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# plt.imshow(img)
# plt.title("Hough Lines")
# plt.show()
rhos = lines[:, 0, 0]
thetas = lines[:, 0, 1]

verticals = []
horizontals = []
intersections = []
angle_thresh = np.pi/18  # within 10 degrees
for rho, theta in zip(rhos, thetas):
    if abs(theta) < angle_thresh or abs(theta - np.pi) < angle_thresh:
        verticals.append((rho, theta))
    elif abs(theta - np.pi/2) < angle_thresh:
        horizontals.append((rho, theta))

h, w = img.shape[:2]
cx, cy = w / 2, h / 2

num_vert, num_horiz = 30, 30
v_lines = sorted(verticals, key = lambda line: abs(line[0] - cx))
v_lines = v_lines[:num_vert]
h_lines = sorted(horizontals, key = lambda line: abs(line[0] - cy))
h_lines = h_lines[:num_horiz]

for line in v_lines:
    rho, theta = line
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
    x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
    
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

for line in h_lines:
    rho, theta = line
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
    x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
    
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.imshow(img)
plt.title("Hough Lines")
plt.show()

for vrho, vtheta in v_lines:
    for hrho, htheta in h_lines:
        A = np.array([[np.cos(vtheta), np.sin(vtheta)],
                     [np.cos(htheta), np.sin(htheta)]])
        b = np.array([vrho, hrho])

        if np.abs(np.linalg.det(A)) < 1e-6:
            continue
        
        x, y = np.linalg.solve(A, b)
        intersections.append((x, y))

intersections = np.array(intersections)
h, w = img.shape[:2]
in_bounds = (intersections[:, 0] >= 0) & (intersections[:, 0] < w) & (intersections[:, 1] >= 0) & (intersections[:, 1] < h)
intersections = intersections[in_bounds]

# kmeans = KMeans(n_clusters=2).fit(intersections)
# labels = kmeans.labels_

# h, w = img.shape[:2]
# centre = np.array([w/2, h/2])
# centroids = kmeans.cluster_centers_

# dists = np.linalg.norm(centroids - centre, axis=1)
# target_cluster = np.argmin(dists)
# inner_points = intersections[labels == target_cluster]

# print(intersections)
# for xi, yi in intersections:
#     cv2.circle(img, (int(xi), int(yi)), 10, (0, 255, 0), -1)
# print(intersections.shape)
rect = cv2.minAreaRect(intersections)
box = cv2.boxPoints(rect)
box = np.intp(box)

pts = box[np.argsort(box[:, 0])]
left = pts[:2]
right = pts[2:]

left = left[np.argsort(left[:, 1])]
tl, bl = left

right = right[np.argsort(right[:, 1])]
tr, br = right

widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)

maxWidth = int(max(widthA, widthB))

heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxHeight = int(max(heightA, heightB))

dst = np.array([[0, 0], [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight -1]], dtype=np.float32)
M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype='float32'), dst)


warped = cv2.warpPerspective(equalised, M, (maxWidth, maxHeight))
cropped_img = warped
cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
cv2.imshow("", warped)
cv2.waitKey()

cropped_img = cv2.adaptiveThreshold(cropped_img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=5, C=2)
import skimage
cropped_img = 255 - cropped_img

cropped_img = skimage.morphology.label(cropped_img, connectivity=2)
cropped_img = skimage.morphology.remove_small_objects(cropped_img, min_size=20)

plt.imshow(cropped_img, cmap='gray')
plt.show()

