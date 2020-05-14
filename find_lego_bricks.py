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

def order_corners( points ):
    # It will first sort on the y value and if that's equal then it will sort on the x value
    ordered = [];
    padding = 10;
    last_y = points[0][1];
    
    points = sorted(points , key=lambda y: y[1]);
    
    row = [];
    
    for i, point in enumerate(points):
        
        # Check if y is out of range
        if(point[1] > last_y - padding  and point[1] < last_y + padding ):
            # Is within range
            row.append( point );
            #print(row);
            
        else:

            row = sorted(row , key=lambda x: x[0]);
            ordered.extend(row);
            
            row = [];
            last_y = point[1];
            
            row.append( point );
            
        if( len(points)-1 == i ):
            row = sorted(row , key=lambda x: x[0]);
            ordered.extend(row);
        
    return ordered;

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

# Map from one variable to another
def pixelMap( x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;

def detect_lego_pos( flatImage ):
    # NOW find contours of lego bricks on flat image
    grayImage = cv2.cvtColor(flatImage, cv2.COLOR_BGR2GRAY)
    # Blur image a little to remove very sharp edges
    blur = cv2.blur(grayImage,(5,5));

    # Convert to binary image (black/white)
    (thresh, binaryImage) = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)

    # Erode to get closer to actual border 
    kernel = np.ones((5,5))
    erosion = cv2.erode(binaryImage, kernel,iterations = 1)

    # Invert
    inverted = cv2.bitwise_not(binaryImage);

    # Find contours in inverted binary image
    (contours,hier) = cv2.findContours(erosion,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE);

    # Creating empty array
    polygones = [];
    corners = [];
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
        new = new.astype(int);
        
        # Save polygone and 1st corner of polygone for later use
        polygones.append(new);
        corners.append(new[0]);
     
        h_total = h_total + new[3][1] - new[0][1];
        h_block = h_total / (i+1);
    
    corners = order_corners(corners);
    
    return corners, int(h_block); 

mean_color = {
    "blue":   [100, 250],
    "red":    [0, 250],
    "yellow": [30, 250],
    "orange": [15, 250],
    "green":  [60, 250],
    "white":  [0, 0]
}

def detect_lego_color( flatImage, corners, height ):

    # Loop though all polygones, to determine the color within this area
    for i, corner in enumerate(corners):
        
        # The first point in the polygone, is the upper left corner
        top = corner;

        x = top[0];
        y = top[1];

        # Select area to check
        cropped = flatImage[y:y+height, x:x+height]
        
        # Convert cropped to hsv to check for hue and saturation
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV);
        h, s, v = cv2.split(hsv);
        
        # Merge hue and saturation, to detect difference between red and white (has same hue)
        hsv = cv2.merge((h, s));

        for color in mean_color:
            lower = np.array(mean_color[color]) - 5;
            upper = np.array(mean_color[color]) + 5;

            mask = cv2.inRange(hsv, lower, upper);
            
            if 255 in mask:
                print(i, color);

    
# Load image taken from roboDK 2d camera
filename = 'test.jpg'
image = cv2.imread(filename)


flatImage, plate_dim = get_flat_plate(image);
    
corners, block_height = detect_lego_pos( flatImage );

detect_lego_color( flatImage, corners, block_height );


# Draw each lego polygone
for i, corner in enumerate(corners):
    cv2.putText(flatImage, str(i), tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(flatImage, tuple(corner), tuple(corner + block_height), (0,255,0), 2)
    
# Selecting a brick
brick_id = 0;
brick = corners[brick_id];

cv2.circle(flatImage, tuple(brick), 5, [255,255,255], 2)
x = brick[0];
y = brick[1];

x_mm = pixelMap(x, 0, plate_dim, 260, 0);
y_mm = pixelMap(y, 0, plate_dim, 0, 260);

print('\nBrick ' + str(brick_id) + ' is located at:')
print( 'abs x', x_mm - 16 + 300, 'mm');
print( 'abs y', y_mm + 16 - 400, 'mm');


#print( h_block );
#print( round(h_block) );



#cv2.imshow('Original',image);
cv2.imshow('Warped with stuff',flatImage);
#cv2.imshow('Binary',binaryImage);

cv2.waitKey(0)
cv2.destroyAllWindows()

