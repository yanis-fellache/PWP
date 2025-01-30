import cv2
import numpy as np


def process_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply a blur to the grayscale image
    # gray = cv2.blur(gray, (4, 4))
    # # Apply a median blur to the grayscale image
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # mask = np.zeros(gray.shape[:2], dtype="uint8")
    # cv2.circle(mask, (1400, 1000), 700, 255, -1)
    # masked = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow("Circles", mask)
    # cv2.waitKey(0)
    # cv2.imshow("Circles", masked)
    # cv2.waitKey(0)

    edges = cv2.Canny(gray, 30, 200)

    return edges


def detect_circles(processed_img):
    # Get the number of rows in the processed image
    rows = processed_img.shape[0]

    # Use HoughCircles to detect circles in the processed image
    circles = cv2.HoughCircles(processed_img, cv2.HOUGH_GRADIENT, dp=1, minDist=rows / 8, param1=50, param2=40)
    return circles


# Function to draw circles on an input image
def draw_circle(img, circles):
    # Check if any circles are detected
    if circles is not None:
        # Convert circle coordinates to integer
        circles = np.uint16(np.around(circles))
        # print(circles)
        circles = sorted(circles[0, :], key=lambda x: x[2], reverse=True)
        circle = circles[0]
        center = (circle[0], circle[1])
        print(f"Center: {int(circle[0])}, {int(circle[1])} Radius: {circle[2]}")
        cv2.circle(img, center, 1, (0, 0, 255), 5)
        rad = circle[2]
        cv2.circle(img, center, rad, (0, 255, 0), 5)

        # Loop through detected circles
        # for circle in circles[0, :]:
        #     center = (circle[0], circle[1])
        #     print(f"Center: {int(circle[0])}, {int(circle[1])} Radius: {circle[2]}")
        #     # Draw a small dot at the center of the circle
        #     cv2.circle(img, center, 1, (0, 0, 255), 10)
        #
        #     # Draw the circle outline
        #     rad = circle[2]
        #     cv2.circle(img, center, rad, (0, 255, 0), 10)

    return img


def main():
    # frame = cv2.imread(sys.argv[1])
    # process_frame = process_image(frame)
    # cv2.imshow("Circles", process_frame)
    # cv2.waitKey(0)
    # circles = detect_circles(process_frame)
    # img_with_circle = draw_circle(frame, circles)
    # cv2.imshow("Circles", img_with_circle)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cap = cv2.VideoCapture('RollingCan2.mov')
    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)

        process_frame = process_image(frame)
        # # cv2.imshow("Circles", process_frame)
        # # cv2.waitKey(0)
        circles = detect_circles(process_frame)
        frame = draw_circle(frame, circles)

        cv2.imshow('Circles', frame)
        # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
