#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
import math
import cv2
#%%
def read_mnist_images(filename):
    with open(filename, 'rb') as file:
        data = np.fromfile(file, dtype=np.uint8, count=-1, sep='')
    return data[16:].reshape(-1, 28, 28)

def read_mnist_labels(filename):
    with open(filename, 'rb') as file:
        data = np.fromfile(file, dtype=np.uint8, count=-1, sep='')
    return data[8:]

train_images_file = 'train-images-idx3-ubyte/train-images.idx3-ubyte'
train_labels_file = 'train-labels-idx1-ubyte/train-labels.idx1-ubyte'
test_images_file = 't10k-images-idx3-ubyte/t10k-images.idx3-ubyte'
test_labels_file = 't10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte'

#reading dataset
train_images = read_mnist_images(train_images_file)
train_labels = read_mnist_labels(train_labels_file)
test_images = read_mnist_images(test_images_file)
test_labels = read_mnist_labels(test_labels_file)
#%%
def histogram(images, num_bins=8):
    features = []
    for image in images:
        hist= np.histogram(image, bins=num_bins, range=(0, 256))
        features.append(hist[0])
    return np.array(features)

def lbp_point(image, x, y):
    lbp_value = []
    list2 = [128,64,32,16,8,4,2,1]
    center = image[x, y]

    coordinates = [(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1) ]

    for x_offset,y_offset in coordinates:
        neighbor_x = x + x_offset
        neighbor_y = y + y_offset

        if image[neighbor_x, neighbor_y] >= center:
            lbp_value.append(1)
        else:
            lbp_value.append(0)
    result = sum(x * y for x, y in zip(lbp_value, list2))
    return result

def lbp_image(image):
    height, width = image.shape
    lbp_image = np.zeros((height - 2 , width - 2 ), dtype=np.uint8)

    for x in range(1, height - 1):
        for y in range(1, width - 1):
            lbp_image[x -1, y -1] = lbp_point(image, x, y)
    hist = np.histogram(lbp_image , bins = np.arange(256))
    return hist[0]
def lbp_data(data):
    lbp_all = []
    for i in data :
        ded = lbp_image(i)
        lbp_all.append(ded)
    return ded

def hog(img):
    img = np.array(img)
    mag = []
    theta = []
    for i in range(128):
        magnitudeArray = []
        angleArray = []
        for j in range(128):
            # Condition for axis 0
            if j-1 <= 0 or j+1 >= 128:
                if j-1 <= 0:
                    # Condition if first element
                    Gx = img[i][j+1] - 0
                elif j + 1 >= len(img[0]):
                    Gx = 0 - img[i][j-1]
                    # Condition for first element
            else:
                Gx = img[i][j+1] - img[i][j-1]
        
            # Condition for axis 1
            if i-1 <= 0 or i+1 >= 128:
                if i-1 <= 0:
                    Gy = 0 - img[i+1][j]
                elif i +1 >= 128:
                    Gy = img[i-1][j] - 0
            else:
                Gy = img[i-1][j] - img[i+1][j]

            # Calculating magnitude
            magnitude = math.sqrt(pow(Gx, 2) + pow(Gy, 2))
            magnitudeArray.append(round(magnitude, 9))

            # Calculating angle
            if Gx == 0:
                angle = math.degrees(0.0)
            else:
                angle = math.degrees(abs(math.atan(Gy / Gx)))
            angleArray.append(round(angle, 9))
        mag.append(magnitudeArray)
        theta.append(angleArray)
    mag = np.array(mag)
    mag_toreturn = np.histogram(mag , bins=np.arange(256))
    mag = np.array(mag)
     

    theta = np.array(theta)

    number_of_bins = 9
    step_size = 180 / number_of_bins
     
    def calculate_j(angle):
        temp = (angle / step_size) - 0.5
        j = math.floor(temp)
        return j

    def calculate_Cj(j):
        Cj = step_size * (j + 0.5)
        return round(Cj, 9)

    def calculate_value_j(magnitude, angle, j):
        Cj = calculate_Cj(j+1)
        Vj = magnitude * ((Cj - angle) / step_size)
        return round(Vj, 9)

    histogram_points_nine = []
    for i in range(0, 128, 8):
        temp = []
        for j in range(0, 128, 8):
            magnitude_values = [[mag[i][x] for x in range(j, j+8)] for i in range(i,i+8)]
            angle_values = [[theta[i][x] for x in range(j, j+8)] for i in range(i, i+8)]
            for k in range(len(magnitude_values)):
                for l in range(len(magnitude_values[0])):
                    bins = [0.0 for _ in range(number_of_bins)]
                    value_j = calculate_j(angle_values[k][l])
                    Vj = calculate_value_j(magnitude_values[k][l], angle_values[k][l], value_j)
                    Vj_1 = magnitude_values[k][l] - Vj
                    bins[value_j]+=Vj
                    bins[value_j+1]+=Vj_1
                    bins = [round(x, 9) for x in bins]
                temp.append(bins)
        histogram_points_nine.append(temp)
    epsilon = 1e-05
    feature_vectors = []
    for i in range(0, len(histogram_points_nine) - 1, 1):
        temp = []
        for j in range(0, len(histogram_points_nine[0]) - 1, 1):
            values = [[histogram_points_nine[i][x] for x in range(j, j+2)] for i in range(i, i+2)]
            final_vector = []
            for k in values:
                for l in k:
                    for m in l:
                        final_vector.append(m)
                k = round(math.sqrt(sum([pow(x, 2) for x in final_vector])), 9)
                final_vector = [round(x/(k + epsilon), 9) for x in final_vector]
                temp.append(final_vector)
        feature_vectors.append(temp)
     
    
    return mag_toreturn[0],feature_vectors
def hog_data(data):
    hog_all = []
    for i in data :
        ded , vec = hog(i)
        hog_all.append(vec)
    return hog_all

def linear_kernel(X, Y):
    return np.dot(X, Y.T)

def polynomial_kernel(X, Y, degree):
    return (np.dot(X, Y.T) + 1) ** degree

def rbf_kernel(X, Y, sigma):
    #euclidean distances
    pairwise_distances_squared = np.array(np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T),dtype=np.float64)
    return np.exp(-pairwise_distances_squared / (2 * sigma**2))
#%%

train_images_flattened = train_images[:3000].reshape(train_images[:3000].shape[0], -1)
test_images_flattened = test_images[:500].reshape(test_images[:500].shape[0], -1)

train_hoi_features = histogram(train_images)[:3000]
test_hoi_features = histogram(test_images)[:500]

train_lbp_features = lbp_data(train_images[:3000])
test_lbp_features = lbp_data(test_images[:500])
#%%
train_hog_features = hog_data(train_images[:3000])
test_hog_features = hog_data(test_images[:500])
print(train_images_flattened.shape)
print(test_images_flattened.shape)
# %%
