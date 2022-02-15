import numpy as np
import cv2

for i in range(10):
    img = np.zeros([28, 28, 1], 'uint8') + 255
    cv2.putText(img, f"{i}", (3, 24), cv2.FONT_HERSHEY_COMPLEX, 1, 0, 1)
    cv2.imshow("res", img)
    cv2.waitKey(1)
    cv2.imwrite(f'{i}.jpg', img)
