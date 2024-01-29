#%%
import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


#%%

# Function to compute LBP for a single pixel
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

# Function to compute LBP for the entire image
def lbp_image(image):
    height, width = image.shape
    lbp_image = np.zeros((height - 2 , width - 2 ), dtype=np.uint8)

    for x in range(1, height - 1):
        for y in range(1, width - 1):
            lbp_image[x -1, y -1] = lbp_point(image, x, y)
    hist = np.histogram(lbp_image , bins = np.arange(256))
    return hist[0]
#%%

import pathlib
import tensorflow as tf

n= 0
data_train = []
data_test = []
data_val = []

for i in os.listdir('actors/images'):
    path = 'actors/images/' + i
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
      img_lbp = lbp_image(image)
      csv.append(img_lbp)
  csv = pd.DataFrame(csv)
  csv.to_csv(file_name)
  return csv

#%%
data_train_lbp = csv_images(data_train , 'train_q3.csv')
data_test_lbp = csv_images(data_test , 'test_q3.csv')
data_val_lbp = csv_images(data_val , 'val_q3.csv')

#%%
train_labels = data_train[1]
test_labels = data_test[1]
val_labels = data_val[1]

#%%



from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
#%%

svm = OneVsRestClassifier(SVC(kernel='linear'))
svm.fit(data_train_lbp, np.array(train_labels).flatten())
y_val_pred = svm.predict(data_val_lbp)
accuracy_val = accuracy_score(np.array(val_labels).flatten(), y_val_pred)
print("SVM Validation Accuracy:", accuracy_val)
y_test_pred = svm.predict(data_test_lbp)
accuracy_test = accuracy_score(np.array(test_labels).flatten(), y_test_pred)
print("SVM Test Accuracy:", accuracy_test)


#%%
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(test_labels,y_test_pred, labels=range(1, 7 + 1))
print(confusion)
plt.show()

#%%
n= 0
data_train_2 = []
data_test_2 = []
data_val_2 = []

def read_images(path , cls =0 ):
   
   dir_imgs =[]
   for i in os.listdir(path):
      path_img = path + '/' + i
      img = cv2.imread(path_img , cv2.IMREAD_GRAYSCALE)
      img.resize(128,128)
      dir_imgs.append((img , cls))
   return dir_imgs

data_val_2 = read_images('data_q3_3/val')
data_test_2 = read_images('data_q3_3/test')
data_train_2 = read_images('data_q3_3/train')

data_val_2 = pd.DataFrame(data_val_2)
data_test_2 = pd.DataFrame(data_test_2)
data_train_2 = pd.DataFrame(data_train_2)
data_train_lbp_2 = csv_images(data_train_2 , 'train_q3_2.csv')
data_test_lbp_2 = csv_images(data_test_2 , 'test_q3_2.csv')
data_val_lbp_2 = csv_images(data_val_2 , 'val_q3_2.csv')
#%%

x_train = pd.concat([data_train_lbp,data_train_lbp_2])
y_train =[0] * len(data_train_lbp) + [1] * len(data_train_lbp_2)

x_test = pd.concat([data_test_lbp,data_test_lbp_2])
y_test =[0] * len(data_test_lbp) + [1] * len(data_test_lbp_2)


svm_2 = SVC(kernel='linear')
svm_2 = svm_2.fit(x_train, y_train)
predictions = svm_2.predict(x_test)
print(accuracy_score(y_test , predictions))

