import cv2
import numpy as np

def pyramid(image, scale=1.5, min_size=(40, 40)):
    yield image

    # Generate pyramid levels until minimum size is reached
    while True:
        # Calculate the new image size based on
        # the scale factor and resize the image
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))

        # If the new level is too small, stop generating more levels
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        yield image


def sliding_window(image, step_size, window_size):
    h, w = window_size
    image_h, image_w = image.shape[:2]
    for y in range(0, image_h, step_size):
        for x in range(0, image_w, step_size):
            window = image[y:y + h, x:x + w]
            if window.shape[:2] != window_size:
                continue
            yield x, y, window


def process_image(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(edges, contours, -1, (255, 255, 255), 3)
    return edges


image = cv2.imread("lines.png")
processed_image = process_image(image)
w, h = 156, 156

for resized in pyramid(image):
    for (x, y, window) in sliding_window(resized, step_size=40, window_size=(w, h)):

        lines = cv2.HoughLinesP(processed_image, 1, np.pi / 180, 200, minLineLength=-1, maxLineGap=10)
        for points in lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            print(points[0])
            # Draw the lines joining the points on the original image
            cv2.polylines(image, [np.array([[x1, y1], [x2, y2]], np.int32)], False, (0, 255, 0), 2)

        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(100)