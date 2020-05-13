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
    
    # We know the board is square, so only one of the dimensions are needed
    maxDim = max(maxWidth, maxHeight);
    
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxDim, 0],
        [maxDim, maxDim],
        [0, maxDim]
    ], dtype = "float32")
    
    # print(dst);
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxDim, maxDim))
    # return the warped image
    return warped, maxDim
    
def get_flat_plate( image ):
    
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur image a little to remove very sharp edges
    blur = cv2.blur(grayImage,(3,3));
    
    # Convert to binary image (black/white)
    (thresh, binaryImage) = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
    # Invert
    inverted = cv2.bitwise_not(binaryImage);

    # Find contours in inverted binary image
    (contours,hier) = cv2.findContours(inverted,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE);

    # The first contour is always the most outer contour
    plateContour = contours[0];
    
    # Fill the contour with white, to remove all the lego bricks (only plate left).
    #cv2.drawContours(inverted,[plateContour],0,255,-1)
    
    # Convert back to normal binary
    #binaryImage = cv2.bitwise_not(inverted);

    # Now fit polyline to outer contour
    epsilon = 0.1 * cv2.arcLength(plateContour, True)
    points = cv2.approxPolyDP(plateContour, epsilon, True)

    # Generate coordinates for imageTransformation
    coordinates = np.zeros((4, 2))
    for i, point in enumerate(points):
        coordinates[i] = point[0];

    # Warp the image, to get a flat image of the plate = no perspective
    warped = four_point_transform(image, coordinates);
    
    # Draw polyline (all points) on original image
    #cv2.drawContours(image, [points], -1, (0, 255, 0), 4)
    
    return warped;

#dst = cv2.cornerHarris(binaryImage,2,3,0.04)

#result is dilated for marking the corners, not important
#dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
#image[dst>0.01*dst.max()]=[0,255,255]

#cv2.imshow('Binary',image);
# cv2.imshow('Dst',detected_edges);

# Load image taken from roboDK 2d camera
filename = 'test.jpg'
image = cv2.imread(filename)


flatImage, plate_dim = get_flat_plate(image);
    
# NOW find contours of lego bricks on flat image

grayImage = cv2.cvtColor(flatImage, cv2.COLOR_BGR2GRAY)
# Blur image a little to remove very sharp edges
blur = cv2.blur(grayImage,(5,5));

# Convert to binary image (black/white)
(thresh, binaryImage) = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)

# Erode to get closer to actual border 
kernel = np.ones((5,5))
erosion = cv2.erode(binaryImage, kernel,iterations = 1)

cv2.imshow('erosion',erosion);

# Invert
inverted = cv2.bitwise_not(binaryImage);



# Find contours in inverted binary image
(contours,hier) = cv2.findContours(erosion,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE);

# Creating empty array
polygones = [];
h_block = 0;
h_total = 0;

# Now fit polyline to all 16 contours to find locations of lego bricks
for i, contour in enumerate(contours):
    
    epsilon = 0.1 * cv2.arcLength(contours[i], True)
    polygone = cv2.approxPolyDP(contours[i], epsilon, True)
    
    new = [];
    for point in polygone:
        new.append( [ point[0][0], point[0][1] ] );
    
    # Order points
    new = order_points( np.array( new ) );
    polygones.append(new);
    
 
    h_total = h_total + new[3][1] - new[0][1];
    h_block = h_total / (i+1);

    
    #cv2.circle(flatImage, tuple(new[0]), 5, [255,255,255], 2)
    #cv2.drawContours(flatImage, [polygone], -1, (0, 0, 255), 2)

def pixelMap( x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;

# Draw each lego polygone
for polygone in polygones:
    #print(polygone)
    cv2.rectangle(flatImage, tuple(polygone[0]), tuple(polygone[0] + h_block), (0,255,0), 2)
    
# Selecting a polygone
test = polygones[4][0];
cv2.circle(flatImage, tuple(test), 5, [255,255,255], 2)
x = test[0];
y = test[1];

x_mm = pixelMap(x, 0, plate_dim, 260, 0);
y_mm = pixelMap(y, 0, plate_dim, 0, 260);

print('x',x, 'px');
print('y',y, 'px');
print( 'x', x_mm, 'mm');
print( 'y', y_mm, 'mm');

print( 'Coordinates of lego block 4, with respect to world frame' );
print( 'abs x', x_mm - 16 + 300, 'mm');
print( 'abs y', y_mm + 16 - 400, 'mm');
#print( h_block );
#print( round(h_block) );



cv2.imshow('Original',image);
cv2.imshow('Warped',flatImage);
cv2.imshow('Binary',binaryImage);

cv2.waitKey(0)
cv2.destroyAllWindows()

