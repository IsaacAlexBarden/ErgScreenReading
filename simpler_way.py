import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skimage

path = "C:/Users/isaac/OneDrive/Documents/Coding/PythonCoding/ComputerVisionProjects/Erg Screen Identifier/ErgScreenImages/March25Erg.jpg"
img = cv2.imread(path)
img = cv2.resize(img, (img.shape[1] // 8, img.shape[0] // 8))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.equalizeHist(gray)
ksize = 3
gray = cv2.GaussianBlur(gray, ksize=(ksize, ksize), sigmaX=0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)
img = cv2.adaptiveThreshold(gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=5, C=2)
img = 255 - img

labelled = skimage.morphology.label(img, connectivity=2)
plt.imshow(labelled)
plt.show()
labelled = skimage.morphology.remove_small_objects(labelled, min_size=20)

plt.imshow(labelled, cmap='gray')
plt.show()