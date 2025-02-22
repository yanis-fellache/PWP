import cv2
import numpy as np

img = cv2.imread("path.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([110,50,50], np.uint8)
upper_blue = np.array([130,255,255], np.uint8)

mask = cv2.inRange(hsv, lower_blue, upper_blue)
img_res = cv2.bitwise_and(img, img, mask = mask)


cv2.imshow("Shapes", img_res)

cv2.waitKey(0)
cv2.destroyAllWindows()
