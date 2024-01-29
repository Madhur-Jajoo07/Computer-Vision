#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io, color

path = "n01534433_10256.JPEG"
image = io.imread(path)
gray_image = color.rgb2gray(image)

rows, cols = gray_image.shape
X = np.reshape(gray_image, (rows * cols, 1))

k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
labels = kmeans.labels_

segmented_image = np.reshape(labels, (rows, cols))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(gray_image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Segmented Image (K-Means)")
plt.imshow(segmented_image, cmap="viridis")

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from skimage import io, color

image = io.imread(path)

lab_image = color.rgb2lab(image)

reshaped_image = lab_image.reshape((-1, 3))

num_clusters = 3

gmm = GaussianMixture(n_components=num_clusters, random_state=42)
labels = gmm.fit_predict(reshaped_image)

segmented_image = np.zeros_like(lab_image)
reshaped_labels = labels.reshape(lab_image.shape[:2])

for i in range(num_clusters):
    segmented_image[reshaped_labels == i] = np.mean(lab_image[reshaped_labels == i], axis=0)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(gray_image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Segmented Image (GMM)")
plt.imshow(segmented_image, cmap="viridis")

plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage import io, color

image = io.imread(path)

lab_image = color.rgb2lab(image)

reshaped_image = lab_image.reshape((-1, 3))

bandwidth = estimate_bandwidth(reshaped_image, quantile=0.1, n_samples=500)

meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
labels = meanshift.fit_predict(reshaped_image)

segmented_image = np.zeros_like(lab_image)
reshaped_labels = labels.reshape(lab_image.shape[:2])

for i in range(len(np.unique(labels))):
    segmented_image[reshaped_labels == i] = np.mean(lab_image[reshaped_labels == i], axis=0)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(gray_image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Segmented Image (Mean Shift)")
plt.imshow(segmented_image, cmap="viridis")

plt.show()


# %%
