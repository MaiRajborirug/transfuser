import cv2
import numpy as np
import time


img = np.zeros((1000, 1000, 3))

while True:
    cv2.imshow("img", img)
    if cv2.waitKey(1) == 27:
        break

    img = img + 0.1
    if np.max(img) > 1:
        img = np.zeros((1000, 1000, 3))
    time.sleep(0.1)
    print("bruh")

