import cv2
import numpy as np

dst_path = 'black_white.jpg' # frame of the video
image = cv2.imread(dst_path, cv2.IMREAD_GRAYSCALE)

# Threshold the image to create a binary image
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

# Filter small clusters (adjust the threshold as needed)
min_cluster_size = 100
filtered_labels = np.zeros_like(labels)
for label in range(1, num_labels):
    if stats[label, cv2.CC_STAT_AREA] >= min_cluster_size:
        filtered_labels[labels == label] = label

# Visualize the result
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Clusters', filtered_labels.astype(np.uint8) * 255)
cv2.waitKey(0)
cv2.destroyAllWindows()