#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import cv2

# %%

def gaussian_filter(filter_size, sigma,image):
    kernel = np.zeros((filter_size, filter_size), dtype=np.float32)
    center = filter_size // 2
    total = 0
    for x in range(filter_size):
        for y in range(filter_size):
            kernel[x, y] = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
            total += kernel[x, y]
    kernel /= total
    height, width = image.shape
    padding = kernel.shape[0]
    output = np.zeros_like(image, dtype=np.float32)
    for x in range(padding, height - padding):
        for y in range(padding, width - padding):
            output[x, y] = np.sum(image[x - padding:x + padding + 1, y - padding:y + padding + 1] * kernel)
    return output

def compute_gradients(image):
    gradient_x_matrix = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_y_matrix = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    height, width = image.shape
    gradient_magnitude = np.zeros((height, width), dtype=np.float32)
    gradient_orientation = np.zeros((height, width), dtype=np.float32)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            neighborhood = image[y - 1:y + 2, x - 1:x + 2]

            gradient_x = np.sum(neighborhood * gradient_x_matrix)
            gradient_y = np.sum(neighborhood * gradient_y_matrix)
            gradient_magnitude[y, x] = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
            gradient_orientation[y, x] = np.arctan2(gradient_y, gradient_x)
    return gradient_magnitude, gradient_orientation 


def compute_structure_tensor(gradient_magnitude, gradient_orientation, window_size):
    height, width = gradient_magnitude.shape
    half_window = window_size // 2
    Mxx = np.zeros((height, width), dtype=np.float32)
    Myy = np.zeros((height, width), dtype=np.float32)
    Mxy = np.zeros((height, width), dtype=np.float32)
    for y in range(half_window, height - half_window):
        for x in range(half_window, width - half_window):
            neighborhood_magnitude = gradient_magnitude[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            neighborhood_orientation = gradient_orientation[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            for i in range(window_size):
                for j in range(window_size):
                    weight = neighborhood_magnitude[i, j]
                    angle = neighborhood_orientation[i, j]
                    Mxx[y, x] += (weight * np.cos(angle)) ** 2
                    Myy[y, x] += (weight * np.sin(angle)) ** 2
                    Mxy[y, x] += (weight * np.cos(angle)) * (weight * np.sin(angle))
    return Mxx, Myy, Mxy


def compute_harris_corner_response(Mxx, Myy, Mxy, k):
    det_M = Mxx * Myy - Mxy**2
    trace_M = Mxx + Myy
    harris_response = det_M - k * (trace_M**2)
    return harris_response


def detect_corners(harris_response, threshold):
    height, width = harris_response.shape
    corner_list = []

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if harris_response[y, x] > threshold and \
               harris_response[y, x] > harris_response[y - 1, x - 1] and \
               harris_response[y, x] > harris_response[y - 1, x] and \
               harris_response[y, x] > harris_response[y - 1, x + 1] and \
               harris_response[y, x] > harris_response[y, x - 1] and \
               harris_response[y, x] > harris_response[y, x + 1] and \
               harris_response[y, x] > harris_response[y + 1, x - 1] and \
               harris_response[y, x] > harris_response[y + 1, x] and \
               harris_response[y, x] > harris_response[y + 1, x + 1]:
                corner_list.append((x, y))

    return corner_list
#%%
image = cv2.imread('taj_mahal2.png' ,cv2.IMREAD_GRAYSCALE)
filter_size = 9
sigma = 15.0

blur_image1 = gaussian_filter(filter_size, sigma,image)
gradient_magnitude, gradient_orientation = compute_gradients(blur_image1)
Mxx, Myy, Mxy = compute_structure_tensor(gradient_magnitude, gradient_orientation, 3)
harris_response = compute_harris_corner_response(Mxx, Myy, Mxy, 0.04)
corner_list = detect_corners(harris_response, .1)



# %%

# def non_maximum_suppression(corner_list, harris_response, neighborhood_size):
#     suppressed_corners = []
#     sorted_corners = sorted(corner_list, key=lambda corner: harris_response[corner[1], corner[0]], reverse=True)
#     while sorted_corners:
#         x, y = sorted_corners.pop(0) 
#         suppressed_corners.append((x, y))
#         sorted_corners = [(cx, cy) for cx, cy in sorted_corners if abs(cx - x) > neighborhood_size or abs(cy - y) > neighborhood_size]

#     return suppressed_corners

# suppressed_corners = non_maximum_suppression(corner_list, harris_response, 1)
# plt.imshow(image, cmap='gray')
# x, y = zip(*suppressed_corners)
# plt.scatter(x, y, c='red', s=.5)  
# plt.show()

#%%

plt.imshow(image , cmap='gray')
x, y = zip(*corner_list)
plt.scatter(x, y, c='r', s=.5)
plt.show()
 # %%


points1 = [(117,52),(126,178),(74,228),(62,334),(184,192),(222,47),(216,357),(141,351),(121,260)]
points1 = np.array(points1)
points2 = [(31,23),(112,51),(8,147),(22,278),(103,140),(130,16),(134,293),(66,266),(49,185)]
points2 = np.array(points2)

# %%
import numpy as np


correspondences1 = points1

correspondences2 = points2
def normalize_points(points):
    mean = np.mean(points)
    points -= int(mean)
    scale = np.sqrt(2) / np.std(points)
    return points * scale

correspondences1_normalized = normalize_points(correspondences1)
correspondences1_normalized = np.c_[correspondences1_normalized , np.ones(9)]
correspondences2_normalized = normalize_points(correspondences2)
correspondences2_normalized = np.c_[correspondences2_normalized , np.ones(9)]

# lmao = pd.DataFrame(correspondences1)
# zeros = np.zeros(8)
# zeros = pd.Series(zeros)
# lmao['ded'] = zeros
# correspondences1_normalized = np.
# Create the coefficient matrix for the linear system
A = np.zeros((len(correspondences1), 9))

for i in range(len(correspondences1)):
    x, y, _ = correspondences1_normalized[i]
    x_, y_, _ = correspondences2_normalized[i]
    A[i] = [x_ * x, x_ * y, x_, y_ * x, y_ * y, y_, x, y,1]

_, _, v = np.linalg.svd(A)
E = v[-1].reshape(3, 3)

u, s, vt = np.linalg.svd(E)
s[2] = 0
E = u @ np.diag(s) @ vt

E = np.dot( correspondences2_normalized.T,np.dot(correspondences2_normalized, E))


print("Essential matrix:")
print(E)

# %%

