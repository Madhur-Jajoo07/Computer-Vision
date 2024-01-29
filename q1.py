#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 

def video_capture(vid_path,frame,name):
    ded = cv2.VideoCapture(vid_path)
    a = True
    i = 0 # for frame counts
    while(a):
        a,b = ded.read()
        if(i == frame):
            b = cv2.resize(b,(256,256))
            b=cv2.cvtColor(b, cv2.COLOR_BGR2RGB ) 
            plt.imshow(b)
            plt.savefig(name)
        i += 1
image1 = video_capture('v_Biking_g01_c02.avi',10,'frame10.png')
image2 = video_capture('v_Biking_g01_c02.avi',11,'frame11.png')


def points_available( cordinates, limits):
	x,y = cordinates
	x_limit, y_limit = limits
	return 0 <= x and x < x_limit and 0 <= y and y < y_limit


def optical_flow(image1, image2, window_size, min_quality=0.01):

    maximum_corners = 10000
    min_distance = 0.1
    feature_list = cv2.goodFeaturesToTrack(image1, maximum_corners, min_quality, min_distance)

    w = window_size//2

    image1 = image1 / 255
    image2 = image2 / 255

    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])

    fx = cv2.filter2D(image1, -1, kernel_x)              
    fy = cv2.filter2D(image1, -1, kernel_y)        

    ft = cv2.filter2D(image2, -1, kernel_t) - cv2.filter2D(image1, -1, kernel_t)


    u = np.zeros(image1.shape)
    v = np.zeros(image1.shape)

    for feature in feature_list:        
            j, i = feature.ravel()		
            i, j = int(i), int(j)		

            I_x = fx[i-w:i+w+1, j-w:j+w+1].flatten()
            I_y = fy[i-w:i+w+1, j-w:j+w+1].flatten()
            I_t = ft[i-w:i+w+1, j-w:j+w+1].flatten()

            b = np.reshape(I_t, (I_t.shape[0],1))
            A = np.vstack((I_x, I_y)).T

            U = np.matmul(np.linalg.pinv(A), b)    

            u[i,j] = U[0][0]
            v[i,j] = U[1][0]
 
    return (u,v)



def drawOnFrame(frame, U, V, output_file):

    line_color = (0, 255, 0) 

    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            u, v = U[i][j], V[i][j]

            if u and v:
                frame = cv2.arrowedLine( frame, (i, j), (int(round(i+u)), int(round(j+v))),
                                        (0, 255, 0),
                                        thickness=1
                                    )
    cv2.imwrite(output_file, frame)



def drawSeperately(image1, image2, U, V, output_file):

    displacement = np.ones_like(img2)
    displacement.fill(255.)             
    line_color =  (0, 0, 0)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):

            start_pixel = (i,j)
            end_pixel = ( int(i+U[i][j]), int(j+V[i][j]) )

            if U[i][j] and V[i][j] and points_available( end_pixel, img1.shape ):     
                displacement = cv2.arrowedLine( displacement, start_pixel, end_pixel, line_color, thickness =2)

    figure, axes = plt.subplots(1,3)
    axes[0].imshow(image1, cmap = "gray")
    axes[0].set_title("first image")
    axes[1].imshow(image2, cmap = "gray")
    axes[1].set_title("second image")
    axes[2].imshow(displacement, cmap = "gray")
    axes[2].set_title("displacements")
    figure.tight_layout()
    plt.savefig(output_file, bbox_inches = "tight", dpi = 200)



img1 = cv2.imread("frame10.png")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("frame11.png")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

U, V = optical_flow( img1, img2, 3, 0.05)

img2 = cv2.cvtColor( img2, cv2.COLOR_GRAY2RGB)
drawSeperately(img1, img2, U, V, "Seperate_Result.png")
drawOnFrame(img2, U, V, 'Result.png')
# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

def video_capture(video_path, frame_number, output_name):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()

    # Capture the specified frame
    for i in range(frame_number):
        success, frame = cap.read()

    if success:
        # Resize and convert to RGB
        frame = cv2.resize(frame, (256, 256))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display and save the frame
        plt.imshow(frame_rgb)
        plt.savefig(output_name)
        plt.show()

    cap.release()

def points_available(coordinates, limits):
    x, y = coordinates
    x_limit, y_limit = limits
    return 0 <= x < x_limit and 0 <= y < y_limit

def optical_flow(image1, image2, window_size, min_quality=0.01):
    maximum_corners = 10000
    min_distance = 0.1
    feature_list = cv2.goodFeaturesToTrack(image1, maximum_corners, min_quality, min_distance)

    w = window_size//2

    image1 = image1 / 255
    image2 = image2 / 255

    kernel_x = np.array([[-1, 1], [-1, 1]])
    kernel_y = np.array([[-1, -1], [1, 1]])
    kernel_t = np.array([[1, 1], [1, 1]])

    fx = cv2.filter2D(image1, -1, kernel_x)
    fy = cv2.filter2D(image1, -1, kernel_y)

    ft = cv2.filter2D(image2, -1, kernel_t) - cv2.filter2D(image1, -1, kernel_t)
    u = np.zeros(image1.shape)
    v = np.zeros(image1.shape)

    for feature in feature_list:        
        j, i = feature.ravel()		
        i, j = int(i), int(j)		

        I_x = fx[i-w:i+w+1, j-w:j+w+1].flatten()
        I_y = fy[i-w:i+w+1, j-w:j+w+1].flatten()
        I_t = ft[i-w:i+w+1, j-w:j+w+1].flatten()

        b = np.reshape(I_t, (I_t.shape[0],1))
        A = np.vstack((I_x, I_y)).T

        U = np.matmul(np.linalg.pinv(A), b)    

        u[i,j] = U[0][0]
        v[i,j] = U[1][0]

    return (u, v)

def draw_on_image(image, U, V, output_file):
    color = (0, 255, 0) 

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            u, v = U[i][j], V[i][j]
            if u and v:
                image = cv2.arrowedLine(image, (i, j), (int(round(i+u)), int(round(j+v))),(0, 255, 0),thickness=1)
    cv2.imwrite(output_file, image)

def draw_separately(image1, image2, U, V, output_file):
    displacement = np.ones_like(img2)
    displacement.fill(255.)             
    line_color =  (0, 0, 0)
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):

            start_pixel = (i,j)
            end_pixel = ( int(i+U[i][j]), int(j+V[i][j]) )

            if U[i][j] and V[i][j] and points_available( end_pixel, img1.shape ):     
                displacement = cv2.arrowedLine( displacement, start_pixel, end_pixel, line_color, thickness =2)

    figure, axes = plt.subplots(1,3)
    axes[0].imshow(image1, cmap = "gray")
    axes[0].set_title("first image")
    axes[1].imshow(image2, cmap = "gray")
    axes[1].set_title("second image")
    axes[2].imshow(displacement, cmap = "gray")
    axes[2].set_title("displacements")
    figure.tight_layout()
    plt.savefig(output_file, bbox_inches = "tight", dpi = 200)

video_capture('v_Biking_g01_c02.avi', 10, 'frame10.png')
video_capture('v_Biking_g01_c02.avi', 11, 'frame11.png')

# Read the captured frames
img1 = cv2.imread("frame10.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("frame11.png", cv2.IMREAD_GRAYSCALE)

# Calculate optical flow
U, V = optical_flow(img1, img2, 3, 0.05)

# Draw displacement vectors on separate images
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
draw_separately(img1, img2_rgb, U, V, "Seperate_Result.png")

# Draw displacement vectors on the frame
draw_on_image(img2_rgb, U, V, 'Result.png')
