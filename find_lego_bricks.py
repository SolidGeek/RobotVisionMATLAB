import cv2
import numpy as np
import time

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect;

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped
    

filename = 'test.jpg'
image = cv2.imread(filename)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur image a little to remove very sharp edges
blur = cv2.blur(grayImage,(3,3));

(thresh, binaryImage) = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)


inverted = cv2.bitwise_not(binaryImage);

#cv2.imshow('Inverted',inverted);

# Find contours in inverted binary image
(contour,hier) = cv2.findContours(inverted,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE);

# The first contour is always the most outer contour, so we fill that with white (to remove alllll the lego bricks)
plateContour = contour[0];

cv2.drawContours(inverted,[plateContour],0,255,-1)

# Convert back to normal binary
binaryImage = cv2.bitwise_not(inverted);

#cv2.imshow('Binary',binaryImage);


# Now fit polyline to outer contour
epsilon = 0.1 * cv2.arcLength(plateContour, True)
points = cv2.approxPolyDP(plateContour, epsilon, True)

# Generate coordinates for imageTransformation
coordinates = np.zeros((4, 2))
for i, point in enumerate(points):
    coordinates[i] = point[0];

warped = four_point_transform(image, coordinates);


# Draw polyline (all points) on original image
cv2.drawContours(image, [points], -1, (0, 255, 0), 4)

cv2.imshow('Original',image);
cv2.imshow('Warped',warped);

#dst = cv2.cornerHarris(binaryImage,2,3,0.04)

#result is dilated for marking the corners, not important
#dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
#image[dst>0.01*dst.max()]=[0,255,255]

#cv2.imshow('Binary',image);
# cv2.imshow('Dst',detected_edges);

cv2.waitKey(0)
cv2.destroyAllWindows()

