import cv2
import numpy as np
from line_processing import compute_line, compute_center
from utils import compute_slope, compute_length, update

def process_frame(frame):
    height, width, _ = frame.shape
    crop_size_x = int(width * 3/5)
    crop_size_y = int(height* 3/5)
    start_x = int(width * 1/5)
    start_y = int(height * 1/5)
    cropped_image = frame[start_y:start_y + crop_size_y, start_x:start_x + crop_size_x]
    # cv2.imshow("Cropped Image", cropped_image)
    # cv2.waitKey(0)

    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # lower_blue = np.array([110, 50, 50])
    # upper_blue = np.array([130, 255, 255])
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # res = cv2.bitwise_and(frame, frame, mask=mask)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=60, maxLineGap=20)

    n_lines = []
    centerline = []
    if lines is not None:
        # for line in lines:
            # print(compute_slope(line[0]))
        n_lines = compute_line(lines)

    if n_lines is not None:
        for line in n_lines:
            # print(f"Line= {line}")
            x1, y1, x2, y2 = line
            # if 0.1 > compute_slope(line) > -0.1:
            #     cv2.line(frame, (int(x1 + width*1/5), int(y1 + height*1/5)), (int(x2 + width*1/5), int(y2 + height*1/5)), (0, 0, 255), 10)
            #     continue
            if compute_length(line) > 200:
                cv2.line(frame, (int(x1 + width*1/5), int(y1 + height*1/5)), (int(x2 + width*1/5), int(y2 + height*1/5)), (0, 255, 0), 10)
                continue
        centerline = compute_center(n_lines)

    # if centerline and compute_slope(centerline) != 0:
    if centerline:
        x1, y1, x2, y2 = centerline
        cv2.line(frame, (int(x1 + width*1/5), int(y1 + height*1/5)), (int(x2 + width*1/5), int(y2 + height*1/5)), (255, 0, 0), 10)

    return frame