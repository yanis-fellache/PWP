import cv2
from image_processing import process_frame

'''Original code from Insomnia Robot Line Detection and adapted for assignment'''

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # frame = cv2.imread("vertical.jpg")
        frame = process_frame(frame)
        cv2.imshow("Image With Lines", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break


    cap.release()
        # cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
