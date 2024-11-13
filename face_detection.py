import cv2
import numpy as np
import sys

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

image1 = cv2.imread(f"{sys.argv[1]}")
image2 = cv2.imread(f"{sys.argv[2]}")

imgray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('Original Image', image1) 
cv2.waitKey(0) 
  
# Gaussian Blur 
Gaussian = cv2.GaussianBlur(image1, (7, 7), 0) 
cv2.imshow('Gaussian Blurring', Gaussian) 
cv2.waitKey(0)

contours = cv2.drawContours(image1, contours, -1, (0,255,0), 3)
cv2.imshow('Contours', contours)
cv2.waitKey(0)

def analyze_face(img):
    faces = face_cascade.detectMultiScale(imgray, 1.3, 5)
    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detects eyes of different sizes in the input image
        eyes = eye_cascade.detectMultiScale(roi_gray) 

        #To draw a rectangle in eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

    # Display an image in a window
    cv2.imshow('Face Analysis',img)
    cv2.waitKey(0)

analyze_face(image1)

cv2.destroyAllWindows()
