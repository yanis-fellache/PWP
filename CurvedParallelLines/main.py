import cv2
import numpy as np


def main():
    # cap = cv2.VideoCapture(0)
    #
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    #
    # cap.release()
    # # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    frame = cv2.imread("lines.png")

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(edges, contours, -1, (255, 255, 255), 3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=-1, maxLineGap=10)
    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        print(points[0])
        # Draw the lines joining the points on the original image
        cv2.polylines(frame, [np.array([[x1, y1], [x2, y2]], np.int32)], False, (0, 255, 0), 2)
    cv2.imshow("Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
