# OpenCV tracking demo, based on https://github.com/colinlaney/animal-tracking
import cv2
import os
import numpy as np


# load a video file
os.chdir('filenamepathhere')
filename = "marker.avi"
cap = cv2.VideoCapture(filename)
ret, frame = cap.read()
cv2.imshow("first frame",frame)


# get the "background" by averaging all the frames (assumes camera doesn't move)
background = frame.copy()
i_frame = 1
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while frame is not None:
        ret, frame = cap.read()
        if frame is None:
            break
        #background = cv2.addWeighted(frame, 0.5, background, 0.5, 0)
        background = cv2.addWeighted(frame, 0.5 * (1 - i_frame / n_frames),
                                background, 0.5 * (1 + i_frame / n_frames), 0)
        i_frame += 1
        
cv2.imshow("background",background)


# now try drawing an example frame with the background subtracted
cap = cv2.VideoCapture(filename)
ret, frame = cap.read()
i_frame = 1
FRAME_NUMBER_OF_INTEREST = 50
while frame is not None:
    if i_frame == FRAME_NUMBER_OF_INTEREST :
        break
    ret, frame = cap.read()        
    if frame is None:   # not logical
        break
    i_frame += 1        

frame1 = cv2.subtract(frame, background)
cv2.imshow("difference between frame "+str(FRAME_NUMBER_OF_INTEREST)+
           " and the background",frame1)



THRESHOLD = 40
frame1Gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
retval, binaryFrame1 = cv2.threshold(frame1Gray, THRESHOLD, 255, cv2.THRESH_BINARY)
cv2.imshow("binary",binaryFrame1)
contours, hierarchy = cv2.findContours(binaryFrame1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


# Find a contour with the biggest area (animal most likely)
contour = contours[np.argmax(list(map(cv2.contourArea, contours)))]
M = cv2.moments(contour)
x = int(M['m10'] / M['m00'])
y = int(M['m01'] / M['m00'])

hull = cv2.convexHull(contour)
imgPoints = np.zeros(frame.shape,np.uint8)
for i in range(2, len(hull) - 2):
       if np.dot(hull[i][0] - hull[i-2][0], hull[i][0] - hull[i+2][0]) > 0:
            imgPoints = cv2.circle(imgPoints, (hull[i][0][0],hull[i][0][1]), 5, BGR_COLOR['yellow'], -1, cv2.LINE_AA)

# Draw a contour of the “animal”
imageWithContour = cv2.drawContours(imgPoints, [contour], 0, (127,255,0), 2, cv2.LINE_AA)
cv2.imshow("contour",imageWithContour)
imgPoints = cv2.circle(imgPoints, (x,y), 5, (0,0,0), -1)

print("Now printing centroid of the biggest contour found")
print("row=",y)
print("col=",x)

