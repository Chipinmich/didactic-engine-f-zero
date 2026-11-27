import cv2

path = r"C:\Users\chipw\fzero-ai\frames\frame.png"

frame = cv2.imread(path)

if frame is None:
    print("Could not load image:", path)
else:
    print("Image loaded:", frame.shape)

    cv2.imshow("F-Zero", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()