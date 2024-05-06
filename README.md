# images_codess
import cv2


import numpy as np

from google.colab.patches import cv2_imshow

i = cv2.imread('gidha.jpg')

cv2_imshow(i)
#cv2.waitkey(0)

gray = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)

cv2_imshow(gray)


#BGR image
ib = i[:,:,0]  # the first page B matrix
ig = i[:,:,1]  # second page G matrix
ir = i[:,:,2]  # third page R matrix
all3 = np.hstack((ib,ig,ir))
cv2_imshow(all3)


temp = i
temp[:,:,2]=0
cv2_imshow(I)

resizeimg = cv2.resize(i,(256,256))
cv2_imshow(resizeimg)
print(resizeimg.shape)


gflip = cv2.flip(resizeimg,1) #0,1,-1
cv2_imshow(gflip)


gcrop = gray[100:200,200:350]
cv2_imshow(gcrop)


import cv2
import numpy as np
import matplotlib.pyplot as plt

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

image = cv2.imread('gidha.jpg')
sharpened_image = sharpen_image(image)
original_and_sharpened_image = np.hstack((image, sharpened_image))
plt.figure(figsize = [30, 30])
plt.axis('off')
plt.imshow(original_and_sharpened_image[:,:,::-1])

def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

image = cv2.imread('dog.png')
sharpened_image = sharpen_image(image)

i = np.hstack((image, sharpened_image))
cv2_imshow(i)

