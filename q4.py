#%%
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from skimage.transform import resize
import math
from skimage.feature import hog
import numpy as np
import cv2
import os
import pandas as pd


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


#%%


n= 0
data_train = []
data_test = []
data_val = []

for i in os.listdir('human detection dataset'):
    path = 'human detection dataset/' + i
    dir_imgs = []
    for j in os.listdir(path):
        path_img = path + '/' + j
        img = cv2.imread(path_img , cv2.IMREAD_GRAYSCALE)
        img.resize(128,128)
        dir_imgs.append(img)
    tot = len(dir_imgs)
    val_len = int(tot*0.2)
    test_len = int(tot*0.1)
    val = dir_imgs[:val_len]
    test = dir_imgs[-test_len:]
    train = dir_imgs[val_len:-test_len]
    for i in train:
        data_train.append((i,n))
    for i in test:
        data_test.append((i,n))
    for i in val:
        data_val.append((i,n))
    n = n+1
data_val = pd.DataFrame(data_val)
data_test = pd.DataFrame(data_test)
data_train = pd.DataFrame(data_train)

#%%
from csv import writer



def csv_images(data , file_name):
  csv = []
  for i in range(data.shape[0]):
      tuple = data.loc[i]
      image = tuple[0]
      img_hog, vector = hog(image)
      csv.append(vector)
  csv = pd.DataFrame(csv)
  csv.to_csv(file_name)
  return csv

#%%
train_vector = csv_images(data_train , 'train_q4.csv')
test_vector = csv_images(data_test , 'test_q4.csv')
val_vector = csv_images(data_val , 'val_q4.csv')

#%%

train_labels = data_train[1]
test_labels = data_test[1]
val_labels = data_val[1]

#%%



from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
#%%

svm = SVC(kernel='linear')
svm = svm.fit(train_vector, np.array(train_labels).flatten())
predictions = svm.predict(val_vector)
accuracy_val = accuracy_score(np.array(val_labels).flatten(), predictions)
print("SVM Validation Accuracy:", accuracy_val)
y_test_pred = svm.predict(test_vector)
accuracy_test = accuracy_score(np.array(test_labels).flatten(), y_test_pred)
print("SVM Test Accuracy:", accuracy_test)




# %%
