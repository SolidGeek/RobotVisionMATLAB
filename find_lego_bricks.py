import cv2
import numpy as np

filename = 'test.jpg'
image = cv2.imread(filename)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

(thresh, binaryImage) = cv2.threshold(grayImage, 1, 255, cv2.THRESH_BINARY)

#dst = cv2.cornerHarris(binary,2,3,0.04)

#result is dilated for marking the corners, not important
#dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
#img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('Gray',grayImage);
cv2.imshow('Binary',binaryImage);

cv2.waitKey(0)
cv2.destroyAllWindows()