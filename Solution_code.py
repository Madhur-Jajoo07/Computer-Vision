#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
import cv2
import gzip
import csv
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score


#%%

# **********Question 1 ********** #

#img_name = 'imagenet_sample_n01622779_10'
img_name = 'imagenet_sample_n01443537_10034'
img = cv2.imread(img_name+".JPEG")
img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)  #converting image to RGB mode from BGR for matplotlib.pyplot
image = img   # making a copy image 
image_gray = cv2.resize(image , (256,256))   #resizing image to 256,256

# Making the image Gray by taking the average of the all three channels i.e. R,G,B.
image_gray = (image_gray[::,::,0:1]*(1/3) + image_gray[::,::,1:2]*(1/3) + image_gray[::,::,2:3]*(1/3) )

#plotting as asked

plt.figure()
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(img)
plt.subplot(1,2,2)
plt.title("Gray image resized to (256,256)")
plt.imshow(image_gray,cmap='gray')
plt.show()
plt.imshow(image_gray,cmap='gray')
plt.savefig(img_name+'.jpg')
# %%

#vertical flip using slicing
flipped_image_vertical = img[::-1]

#horizontal flip using slicing
flipped_image_horizontal =  [row[::-1] for row in img]
#plotting as asked
plt.figure()
plt.subplot(1,3,1)
plt.title("Horizontal Flip")
plt.imshow(flipped_image_horizontal)
plt.subplot(1,3,2)
plt.title("Original")
plt.imshow(img)
plt.subplot(1,3,3)
plt.title("Vertical Flip")
plt.imshow(flipped_image_vertical)
plt.show()

# %%

#generating random numbers x,y to crop image 
img_x = img.shape[0]
img_y = img.shape[1]
x = np.random.randint(64,img_x-64)
y = np.random.randint(64 , img_y-64)

# cropping image using slicing
cropped_image = img[x-64:x+64 , y-64:y+64]
cropped_image = cv2.resize(cropped_image,(256,256)) # resizing
#plotting as asked
plt.figure()
plt.subplot(1,2,1)
plt.title("original Image")
plt.imshow(img)
# plotting the rectangle
x_rect = [x-64, x+64, x+64, x-64, x-64 ]  
y_rect = [y-64, y-64, y+64, y+64, y-64 ]  
plt.plot(y_rect,x_rect)
plt.subplot(1,2,2)
plt.title("Cropped Image")
plt.imshow(cropped_image)
# %%
# **********Question 2 ********** #

def video_capture(vid_path):
    k = int(input("enter the value of k for video Frames :"))
    dir = vid_path[:-4] +"/"
    if not  os.path.exists(dir):  # checking if the directory exists or not if not making one.
        os.makedirs(dir)
    ded = cv2.VideoCapture(vid_path)
    a = True
    i = 0 # for frame counts
    while(a):
        a,b = ded.read()
        path = dir+str(i)+'.jpg'
        if(i%k == 0 ):
            b = cv2.resize(b,(256,256))
            plt.imshow(b)
            plt.savefig(path)
        i += 1

vid_path_1 = 'person15_boxing_d1_uncomp.avi'
video_capture(vid_path_1)

vid_path_2 = 'v_Rowing_g25_c03.avi'
video_capture(vid_path_2)



# %%

def read_mnist_images(images_path):
    with gzip.open(images_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        images_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
    return images_data  

def read_mnist_labels(file_path,x):
    f = gzip.open(file_path,'r')
    f.read(8)
    ded = []
    for i in range(0,x):   
        buf = f.read(1)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        ded.append(labels)
    return ded






# %%
train_images_mnist = read_mnist_images('mnist/train-images-idx3-ubyte.gz')
test_images_mnist = read_mnist_images('mnist/t10k-images-idx3-ubyte.gz')
train_images_mnist_lable = read_mnist_labels('mnist/train-labels-idx1-ubyte.gz',60000)
test_images_mnist_lable = read_mnist_labels('mnist/t10k-labels-idx1-ubyte.gz',10000)

# %%
h = [0]*256     
with open ('train.csv' , 'a+',newline='') as file :
    for img in train_images_mnist: 
        h = [0]*256                 
        for x in range(img.shape[0]):        
            for y in range(img.shape[1]):            
                i = img[x,y]                  
                h[i] = h[i]+1
        
        csv.writer(file , delimiter=',').writerow(h)


# %%
h = [0]*256     
with open ('test.csv' , 'a+',newline='') as file :
    for img in test_images_mnist: 
        h = [0]*256                 
        for x in range(img.shape[0]):        
            for y in range(img.shape[1]):            
                i = img[x,y]                  
                h[i] = h[i]+1
        
        csv.writer(file , delimiter=',').writerow(h)
# %%
def normalize_0_1(data):
    #data = pd.read_csv(path, header=None)
    data = data.reshape(len(data),784)
    means = np.mean(data,axis =0 )
    standard_deviations = np.std(data,axis = 0)
    epsilon = 1e-8
    standard_deviations = np.maximum(standard_deviations, epsilon)
    normalized_data = (data - means) / standard_deviations
    return normalized_data


data_train_normalize = normalize_0_1(train_images_mnist)
data_train_normalize = np.append(data_train_normalize, train_images_mnist_lable, axis=1)
data_test_normalize = normalize_0_1(test_images_mnist) 
data_test_normalize = np.append(data_test_normalize, test_images_mnist_lable, axis=1)

# %%
# %%
tsne = TSNE(n_components=2, perplexity=100, random_state=42)
tsne_result = tsne.fit_transform(train_images_mnist.reshape(60000 , (train_images_mnist[1].shape[0])*(train_images_mnist[1].shape[1])))
#%%
plt.figure(figsize=(10, 8))
s = plt.scatter(tsne_result[:, 0], tsne_result[:, 1],c=train_images_mnist_lable, cmap='rainbow' ,s=5)
labels = ['0','1','2','3','4','5','6','7','8','9']
plt.legend(s.legend_elements()[0], labels)
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()

# %%
def least_square(x,y, lmda):
    xT = np.transpose(x)
    xTx = np.dot(xT,x)
    xTx_lambda_i = xTx + lmda * np.identity(xTx.shape[0])
    xTy = np.dot(xT , y)
    weights = np.dot(np.linalg.inv(xTx_lambda_i), xTy)
    return weights
def least_square_predict(x,weights):
    to_return = []
    for i in x:
        y = np.dot(i,weights)
        to_return.append(np.argmax(y))
    return to_return
# %%
w = least_square(data_train_normalize[:,:-1] ,data_train_normalize[:,-1:],5)
ded = least_square_predict(data_test_normalize[:,:-1] , w)
# %%

two_class_x_train = []
two_class_y_train = []
two_class_x_test = []
two_class_y_test = []

classes = np.random.randint(0,9,2)
for i in data_train_normalize :
    if i[784] == classes[0] :
        two_class_x_train.append(i[:-1])
        two_class_y_train.append([1,0])
    if i[784] == classes[1] :
        two_class_x_train.append(i[:-1])
        two_class_y_train.append([0,1])     
for i in data_test_normalize :
    if i[784] == classes[0] :
        two_class_x_test.append(i[:-1])
        two_class_y_test.append(0)
    if i[784] == classes[1] :
        two_class_x_test.append(i[:-1])
        two_class_y_test.append(1)    

# %%
w = least_square(two_class_x_train ,two_class_y_train , 10)
y_predict_2_class = least_square_predict(two_class_x_test , w)
print("accuracy score for class", classes[0] ,"and", classes[1] , "is : " , 100*round(accuracy_score(y_predict_2_class,two_class_y_test),3) , "%")



# %%
y=[[int(x[0]==0),int(x[0]==1),int(x[0]==2),int(x[0]==3),int(x[0]==4),int(x[0]==5),int(x[0]==6),int(x[0]==7),int(x[0]==8),int(x[0]==9)] for x in train_images_mnist_lable]
w = least_square(data_train_normalize[:,:-1] ,y,5)
y_predict_10_class = least_square_predict(data_test_normalize[:,:-1] , w)
print("accuracy score for all the 10 classes is :", 100*round(accuracy_score(y_predict_10_class,test_images_mnist_lable),3) ,"%")



# %%
data_train_x = pd.read_csv('train.csv',header = None)
data_train_y = train_images_mnist_lable
data_test_x = pd.read_csv('test.csv',header=None)
data_test_y = test_images_mnist_lable
data_train_mean = np.mean(data_train_x , axis = 0)
data_train_x = data_train_x - data_train_mean
data_test_x= data_test_x - data_train_mean

covariance_matrix = np.cov(data_train_x, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# %%
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
top_eigenvectors = eigenvectors[:, :2]
pca = np.dot(data_train_x, top_eigenvectors)
# %%
plt.figure(figsize=(10, 6))
plt.scatter(pca[:,0], pca[:,1], c=data_train_y, cmap='viridis', edgecolor='k')
plt.title('PCA Results: PC1 vs PC2')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Label')
plt.show()
#%%
test_pca = np.dot(data_test_x, top_eigenvectors)

plt.figure(figsize=(10, 6))
plt.scatter(test_pca[:,0], test_pca[:,1], c=data_test_y, cmap='viridis', edgecolor='k')
plt.title('Test Data in PCA Space: PC1 vs PC2')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
# %%
