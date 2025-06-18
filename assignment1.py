
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

#only works for the grayscale images
def Reduce_intensity(image, intensity_size):
    factor = 256 // intensity_size
    reduce_image = (image // factor) * factor
    return reduce_image

def Average_of_image(image,keneral_size):
    return cv2.blur(image,(keneral_size,keneral_size))

def Img_rotate(image, angle):
    (h,w) = image.shape[:2] # get first two value of image.shape
    center = (w//2,h//2) # find the center point of the image 
    matrix2D = cv2.getRotationMatrix2D(center=center,angle=angle, scale=1.0) #make a matrix rotation angle about thr center
    rotated_img = cv2.warpAffine(image,matrix2D,(w,h)) #implement an image according to the matrix2D
    return rotated_img

def block_average(image, block_size):
    h, w = image.shape[:2]
    h_crop = h - h % block_size
    w_crop = w - w % block_size
    image_cropped = image[:h_crop, :w_crop]
    
    out = image_cropped.reshape(h_crop // block_size, block_size, w_crop // block_size, block_size, -1).mean(axis=(1, 3)) #Computes the mean of each block using mean(axis=(1, 3))
    return np.kron(out.astype(np.uint8), np.ones((block_size, block_size, 1))) #repeats each block value to recreate the full-size pixelated image.



#Task 1: To reduce the number of intensity levels in an image from 256 to 2
image1 = cv2.imread("aston.JPEG") # input the image using cv2
image_gray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY) # make the source input image RGB to GRAY scale
intensity = int(input("Enter the intensity level you want: "))
reduced_image = Reduce_intensity(image_gray,intensity)
#Task 1 outputs
cv2.imwrite(f"reduced_{intensity}_levels.png", reduced_image)


#Task 2: perform a simple spatial 3x3, 10x10 and 20x20 average of image pixels
average_3 = Average_of_image(image1 ,3)
average_10 = Average_of_image(image1,10)
average_20 = Average_of_image(image1,20)
#Task 2 outputs
cv2.imwrite("average_3x3.png", average_3)
cv2.imwrite("average_10x10.png", average_10)
cv2.imwrite("average_20x20.png", average_20)


#Task 3: Rotate the image 45 and 90 angles
angle45 = Img_rotate(image1,45)
angle90 = Img_rotate(image1,90)
#Task 3 outputs
cv2.imwrite("rotated_45_degree.png", angle45)
cv2.imwrite("rotated_90_degree.png", angle90)


#Task 4:  reducing the image spatial resolution.
block_3 = block_average(image1, 3)
block_5 = block_average(image1, 5)
block_7 = block_average(image1, 7)
#Task 4 outputs
cv2.imwrite("block_3.png", block_3)
cv2.imwrite("block_5.png", block_5)
cv2.imwrite("block_7.png", block_7)
