import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

'''
Add Calculate Slope function and extend detected lines
Find a way to remove grass from vision
Detect Arrows and add Compas Rose
'''

plt.ion()

def perspective(img):
    tl, bl, tr, br = (130, 350), (10, 400), (460, 350), (530, 400)
    # cv2.circle(img, tl, 5, (0, 0, 255), -1)
    # cv2.circle(img, bl, 5, (0, 0, 255), -1)
    # cv2.circle(img, tr, 5, (0, 0, 255), -1)
    # cv2.circle(img, br, 5, (0, 0, 255), -1)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [480, 640]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    warped = cv2.warpPerspective(img, matrix, (640, 480))
    cv2.imshow("Warped", warped)
    return warped, inv_matrix


def color_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(hsv, (0, 0, 200), (255, 30, 255))
    yellow_mask = cv2.inRange(hsv, (10, 70, 80), (50, 255, 255))
    green_mask = cv2.inRange(hsv, (25, 40, 40), (90, 255, 255))
    non_grass_mask = cv2.bitwise_not(green_mask)

    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
    final_mask = cv2.bitwise_and(lane_mask, non_grass_mask)
    masked = cv2.bitwise_and(img, img, mask=final_mask)

    cv2.imshow("Masked (No Grass)", masked)
    return masked


def process_image(img):
    masked = color_mask(img)
    blur = cv2.GaussianBlur(masked, (3, 3), 1)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # cv2.imshow("Proccessed", edges)
    return edges


def compute_slope(line):
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:
        return 0
    return (y2 - y1) / (x2 - x1)


def hough_lines(img):
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=200)
    return lines


def get_histogram(img):
    return np.sum(img[img.shape[0]//2:, :], axis=0)


def plot_histogram(histogram):
    plt.clf()
    plt.plot(histogram)
    plt.title("Lane Line Histogram")
    plt.xlabel("Pixel Position")
    plt.ylabel("Sum of Pixel Intensities")
    plt.pause(0.1)


def calculate_centerline(left_curve, right_curve, plot_y):
    center_curve = (left_curve + right_curve) / 2
    return np.array([np.transpose(np.vstack([center_curve, plot_y]))])


def sliding_window_search(binary_warped, original_img, inv_matrix):
    histogram = get_histogram(binary_warped)
    # plot_histogram(histogram)
    midpoint = np.int32(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    nwindows = 9
    window_height = np.int32(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    leftx_current, rightx_current = leftx_base, rightx_base
    margin, minpix = 100, 50
    left_lane_inds, right_lane_inds = [], []
    
    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
        win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin
        
        left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                     (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                      (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(left_inds)
        right_lane_inds.append(right_inds)
        
        if len(left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[left_inds]))
        if len(right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[right_inds]))
    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]
    
    lane_image = np.zeros_like(original_img)
    plot_y = np.linspace(0, original_img.shape[0] - 1, original_img.shape[0])
    
    if len(leftx) and len(lefty):
        left_fit = np.polyfit(lefty, leftx, 2)
        left_curve = np.polyval(left_fit, plot_y)
        left_points = np.array([np.transpose(np.vstack([left_curve, plot_y]))])
        cv2.polylines(lane_image, np.int32([left_points]), isClosed=False, color=(0, 255, 0), thickness=10)
    
    if len(rightx) and len(righty):
        right_fit = np.polyfit(righty, rightx, 2)
        right_curve = np.polyval(right_fit, plot_y)
        right_points = np.array([np.transpose(np.vstack([right_curve, plot_y]))])
        cv2.polylines(lane_image, np.int32([right_points]), isClosed=False, color=(0, 0, 255), thickness=10)
    
    if len(leftx) and len(lefty) and len(rightx) and len(righty):
        center_points = calculate_centerline(left_curve, right_curve, plot_y)
        cv2.polylines(lane_image, np.int32([center_points]), isClosed=False, color=(255, 255, 0), thickness=5)
    
    unwarped_lane = cv2.warpPerspective(lane_image, inv_matrix, (original_img.shape[1], original_img.shape[0]))
    result = cv2.addWeighted(original_img, 1, unwarped_lane, 10, 0)
    return result


def arrow_overlay(img, arrow, location):
    arrow = abs(255 - arrow)
    h, w = arrow.shape[:2]
    x, y = location
    shapes = np.zeros_like(img, np.uint8)
    shapes[y: y+h, x: x+w] = arrow
    mask = shapes.astype(bool)
    img[mask] = cv2.addWeighted(img, 1, shapes, 1, 1)[mask]
    return img

def main():
    cap = cv2.VideoCapture('drivingPWP2.mp4')
    arrow = cv2.imread('arrow.png')
    arrow = cv2.resize(arrow, (108, 80))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        p_wrap, inv_matrix = perspective(frame)
        processed_img = process_image(p_wrap)
        frame = sliding_window_search(processed_img, frame, inv_matrix)

        result = arrow_overlay(frame, arrow, (20, 20))

        cv2.imshow("Lane Detection", result)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
