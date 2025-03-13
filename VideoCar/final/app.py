import sys
import time
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

######################################################################################
# Version 4.0
# Program for lane detection and overlaying a compass rose on a video feed.
#
# perspective()
#       Parameters: image frame
#       Return: perspective transformed image and inverse transformation matrix
#
# color_mask()
#       Parameters: image frame
#       Return: color masked image
#
# process_image()
#       Parameters: image frame
#       Return: edge-detected image
#
# compute_avg_distance()
#       Parameters: left lane points, right lane points
#       Return: average lane width
#
# create_points()
#       Parameters: lane points, average distance
#       Return: generated points for missing lanes
#
# compute_average_slope()
#       Parameters: lane x-coordinates, lane y-coordinates
#       Return: average slope of lane
#
# hough_lines()
#       Parameters: edge-detected image
#       Return: detected lines using Hough Transform
#
# get_histogram()
#       Parameters: image frame
#       Return: histogram of pixel intensities in lower half of image
#
# plot_histogram()
#       Parameters: histogram
#       Return: None (displays histogram plot)
#
# calculate_centerline()
#       Parameters: left lane curve, right lane curve, y-coordinates
#       Return: centerline points
#
# arrow_overlay()
#       Parameters: image frame, arrow image, position
#       Return: image with arrow overlayed
#
# sliding_window_search()
#       Parameters: binary warped image, original image, inverse matrix
#       Return: image with detected lanes and centerline
#
# arrow_motion()
#       Parameters: image frame
#       Return: image with rotated directional arrow overlay
#
# main()
#       Parameters: None
#       Return: None (executes video processing loop)
#
######################################################################################


################################## PSEUDOCODE #######################################
#
# 1. Open video file
# 2. Load arrow image and resize
# 3. While video is playing:
#       a. Read frame
#       b. Resize frame
#       c. Apply perspective transformation
#       d. Process image for lane detection
#       e. Detect lanes using sliding window method
#       f. Overlay arrow indicating direction
#       g. Display processed frames
#       h. Exit on key press
# 4. Release video and close all windows
#
######################################################################################

class Application(QtWidgets.QMainWindow):
    def __init__(self, filename):
        super().__init__()

        self.setWindowTitle("Lane Detection")
        self.resize(800, 600)

        self.avg_distance = 250
        self.prev_leftx, self.prev_lefty, self.prev_rightx, self.prev_righty = None, None, None, None
        self.count = 200
        self.angle = 0

        self.__paused = True

        self.__central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.__central_widget)

        self.__layout = QtWidgets.QGridLayout()
        self.__layout.setColumnStretch(0, 1)
        self.__layout.setColumnStretch(1, 1)
        self.__layout.setRowStretch(0, 1)
        self.__layout.setRowStretch(1, 1)

        self.__central_widget.setLayout(self.__layout)

        self.__raw_label = QtWidgets.QLabel(self.__central_widget)
        self.__raw_label.setScaledContents(False)
        self.__raw_label.setAlignment(QtCore.Qt.AlignCenter)
        self.__layout.addWidget(self.__raw_label, 0, 0)

        self.__result_label = QtWidgets.QLabel(self.__central_widget)
        self.__result_label.setScaledContents(False)
        self.__result_label.setAlignment(QtCore.Qt.AlignCenter)
        self.__layout.addWidget(self.__result_label, 1, 0)

        self.__controls_layout = QtWidgets.QGridLayout()
        self.__controls = QtWidgets.QWidget(self.__central_widget)
        self.__controls.setLayout(self.__controls_layout)
        self.__layout.addWidget(self.__controls, 0, 1)

        self.__play_button = QtWidgets.QPushButton(self.__controls)
        self.__play_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPause)
        )
        self.__controls_layout.addWidget(self.__play_button, 1, 1)

        up_arrow = QtWidgets.QPushButton(self.__controls)
        up_arrow.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowUp)
        )
        self.__controls_layout.addWidget(up_arrow, 0, 1)

        down_arrow = QtWidgets.QPushButton(self.__controls)
        down_arrow.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowDown)
        )
        self.__controls_layout.addWidget(down_arrow, 2, 1)

        right_arrow = QtWidgets.QPushButton(self.__controls)
        right_arrow.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowRight)
        )
        self.__controls_layout.addWidget(right_arrow, 1, 2)

        left_arrow = QtWidgets.QPushButton(self.__controls)
        left_arrow.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ArrowLeft)
        )
        self.__controls_layout.addWidget(left_arrow, 1, 0)

        def toggle_playback():
            self.__paused = not self.__paused
            self.__play_button.setIcon(
                self.style().standardIcon(
                    QtWidgets.QStyle.StandardPixmap.SP_MediaPlay if self.__paused else QtWidgets.QStyle.StandardPixmap.SP_MediaPause
                )
            )

        self.__play_button.clicked.connect(toggle_playback)

        self.__log = QtWidgets.QTextEdit(self.__central_widget)
        self.__log.setReadOnly(True)
        self.__log.setFont(QtGui.QFont("monospace"))
        self.__log.setStyleSheet("QTextEdit { padding: 10px; }")
        self.__layout.addWidget(self.__log, 1, 1)

        self.__video_stream = cv2.VideoCapture(filename)
        self.__original_frame = None
        self.__current_frame = None

        self.__arrow_image = cv2.imread("arrow.png", cv2.IMREAD_UNCHANGED)
        self.__arrow_image = cv2.resize(self.__arrow_image, (108, 80))
        self.__frame_count = 0

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.__main_loop)
        self.timer.start()

    def __main_loop(self):
        if self.__paused:
            return

        if not self.begin_frame():
            self.close()
            return

        self.process_frame()
        self.display_frame()
        self.update_log(f"Processed Frame {self.__frame_count}")
        self.__frame_count += 1

    def begin_frame(self) -> bool:
        ret, frame = self.__video_stream.read()
        if not ret or frame is None:
            print("Error: Unable to read frame from video. Ending playback.")
            return False

        self.__original_frame = cv2.resize(frame, (640, 480))
        self.__current_frame = self.__original_frame.copy()
        return True

    def process_frame(self):
        if self.__original_frame is None or self.__original_frame.size == 0:
            return

        p_wrap, inv_matrix = self.perspective(self.__original_frame)

        processed_img = self.process_image(p_wrap)

        sl = self.sliding_window_search(processed_img, self.__original_frame, inv_matrix)

        self.__current_frame = self.arrow_motion(self.__original_frame, sl, self.__arrow_image)

    def display_frame(self):
        raw_frame = cv2.cvtColor(self.__original_frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(self.__current_frame, cv2.COLOR_BGR2RGB)

        raw_image = QtGui.QImage(
            raw_frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888
        )
        self.__raw_label.setPixmap(
            QtGui.QPixmap.fromImage(raw_image).scaled(
                self.__raw_label.width(), self.__raw_label.height(), QtCore.Qt.KeepAspectRatio
            )
        )

        result_image = QtGui.QImage(
            frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888
        )
        self.__result_label.setPixmap(
            QtGui.QPixmap.fromImage(result_image).scaled(
                self.__result_label.width(), self.__result_label.height(), QtCore.Qt.KeepAspectRatio
            )
        )

    def update_log(self, message):
        self.__log.append(message)

    def perspective(self, img):  
        tl, bl, tr, br = (150, 350), (10, 460), (420, 350), (560, 460)
        pts1 = np.float32([tl, bl, tr, br])
        pts2 = np.float32([[0, 0], [0, 480], [640, 0], [480, 640]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
        warped = cv2.warpPerspective(img, matrix, (640, 480))
        return warped, inv_matrix

    def process_image(self, img):
        masked = self.color_mask(img)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    def color_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 10, 200), (255, 20, 255))
        yellow_mask = cv2.inRange(hsv, (10, 70, 80), (50, 255, 255))
        green_mask = cv2.inRange(hsv, (25, 40, 40), (90, 255, 255))
        non_grass_mask = cv2.bitwise_not(green_mask)
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
        final_mask = cv2.bitwise_and(lane_mask, non_grass_mask)
        masked = cv2.bitwise_and(img, img, mask=final_mask)
        return masked

    def compute_avg_distance(self, left_line, right_line):
        total_distance = 0
        num = min(len(left_line), len(right_line))
        for i in range(num):
            total_distance += (right_line[i] - left_line[i])
        current_avg = total_distance / num
        if current_avg > 200: return current_avg
        else: return self.avg_distance

    def create_points(self, line, avgd):
        new_line = []
        for point in line:
            new_line.append(point + avgd)
        return np.array(new_line)

    def compute_average_slope(self, linex, liney):
        total = 0
        for i in range(1, len(linex)):
            total += (liney[i]-liney[i-1])/(linex[i]-linex[i-1])
        return total/len(linex)

    def hough_lines(self, img):
        lines = cv2.HoughLinesP(img, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=200)
        return lines

    def get_histogram(self, img):
        return np.sum(img[img.shape[0] // 2:, :], axis=0)

    def plot_histogram(self, histogram):
        plt.clf()
        plt.plot(histogram)
        plt.title("Lane Line Histogram")
        plt.xlabel("Pixel Position")
        plt.ylabel("Sum of Pixel Intensities")
        plt.pause(0.1)

    def calculate_centerline(self, left_curve, right_curve, plot_y):
        center_curve = (left_curve + right_curve) / 2
        return np.array([np.transpose(np.vstack([center_curve, plot_y]))])

    def sliding_window_search(self, binary_warped, original_img, inv_matrix):  
        histogram = self.get_histogram(binary_warped)
        midpoint = np.int32(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 6
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
            left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                        nonzerox < win_xleft_high)).nonzero()[0]
            right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                        nonzerox < win_xright_high)).nonzero()[0]
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

        if not len(leftx) and not len(lefty) and len(rightx) and len(righty):
            leftx = self.create_points(rightx, -self.avg_distance)
            lefty = righty

        elif not len(rightx) and not len(righty) and len(leftx) and len(lefty):
            rightx = self.create_points(leftx, self.avg_distance)
            righty = lefty

        if len(leftx) and len(lefty) and len(rightx) and len(righty):
            self.avg_distance = self.compute_avg_distance(leftx, rightx)
            avg_slope_left = self.compute_average_slope(leftx, lefty)
            avg_slope_right = self.compute_average_slope(rightx, righty)
            if self.avg_distance < 250:
                leftx, rightx = [], []

        if not len(leftx) and self.prev_leftx is not None:
            leftx, lefty = self.prev_leftx, self.prev_lefty
        if not len(rightx) and self.prev_rightx is not None:
            rightx, righty = self.prev_rightx, self.prev_righty

        if len(leftx):
            self.prev_leftx, self.prev_lefty = leftx, lefty
        if len(rightx):
            self.prev_rightx, self.prev_righty = rightx, righty

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
            center_points = self.calculate_centerline(left_curve, right_curve, plot_y)
            cv2.polylines(lane_image, np.int32([center_points]), isClosed=False, color=(255, 255, 0), thickness=5)
        
        unwarped_lane = cv2.warpPerspective(lane_image, inv_matrix, (original_img.shape[1], original_img.shape[0]))
        result = cv2.addWeighted(original_img, 1, unwarped_lane, 10, 0)
        return result
    
    def arrow_overlay(self, img, arrow, location):  
        h, w, _ = arrow.shape
        x, y = location
        alpha = arrow[:, :, 3] / 255
        blended = (1 - alpha[:, :, None]) * img[y:y + h, x:x + w, :3] + alpha[:, :, None] * arrow[:, :, :3]
        img[y:y + h, x:x + w, :3] = blended.astype(np.uint8)
        return img
    
    def arrow_motion(self, frame, sl, arrow):  
        left = cv2.imread("left.jpg")
        right = cv2.imread("right.jpg")

        if self.count >= 200:
            self.angle = 0

        if frame.shape == right.shape and np.allclose(frame, right, atol=40):
            self.angle = -90
            self.count = 0
        elif frame.shape == left.shape and np.allclose(frame, left, atol=100):
            self.angle = 90
            self.count = 0

        if self.angle != 0:
            (h, w) = arrow.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, self.angle, 1.0)
            rotated_arrow = cv2.warpAffine(arrow, M, (w, h))
        else:
            rotated_arrow = arrow

        self.count += 1
        result = self.arrow_overlay(sl, rotated_arrow, (20, 20))
        return result


def main():
    app = QtWidgets.QApplication(sys.argv)
    if sys.platform == "win32":
        app.setStyle("Fusion")
    widget = Application("drivingPWP.mp4")
    widget.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
