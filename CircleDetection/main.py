import cv2
import numpy as np

# Function to process an input image

'''Code From Hackathon 2024 Group'''


def process_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply a blur to the grayscale image
    gray = cv2.blur(gray, (1, 1))
    # Apply a median blur to the grayscale image
    gray = cv2.medianBlur(gray, 1)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(gray, 30, 200)

    return edges


def detect_circles(processed_img):
    # Get the number of rows in the processed image
    rows = processed_img.shape[0]

    # Use HoughCircles to detect circles in the processed image
    circles = cv2.HoughCircles(processed_img, cv2.HOUGH_GRADIENT, dp=1, minDist=rows / 8, param1=100, param2=140, minRadius=150, maxRadius=1500)
    return circles


# Function to draw circles on an input image
def draw_circle(img, circles):
    # Check if any circles are detected
    if circles is not None:
        # Convert circle coordinates to integer
        circles = np.uint16(np.around(circles))

        # Loop through detected circles
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            print(f"Center: {int(circle[0])}, {int(circle[1])} Radius: {circle[2]}")
            # Draw a small dot at the center of the circle
            cv2.circle(img, center, 1, (0, 0, 255), 10)

            # Draw the circle outline
            rad = circle[2]
            cv2.circle(img, center, rad, (0, 255, 0), 10)

    return img


def main():
    frame = cv2.imread("can.jpg")
    process_frame = process_image(frame)
    # cv2.imshow("Circles", process_frame)
    # cv2.waitKey(0)
    circles = detect_circles(process_frame)
    img_with_circle = draw_circle(frame, circles)
    cv2.imshow("Circles", img_with_circle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

