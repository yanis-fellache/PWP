import cv2
import numpy as np

img = cv2.imread("path.png")
print(img.shape)
img_cropped = img[80:280, 150:330]
cv2.imshow("Shapes", img_cropped)
cv2.waitKey(0)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([110,50,50], np.uint8)
upper_blue = np.array([130,255,255], np.uint8)

mask = cv2.inRange(hsv, lower_blue, upper_blue)
img_res = cv2.bitwise_and(img, img, mask = mask)

canny_image = cv2.Canny(img_res, 100, 200)

lines = cv2.HoughLinesP(
    canny_image,
    rho=6,
    theta=np.pi / 60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25)

line_img = np.copy(canny_image)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 10)

cv2.imshow("Shapes", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
