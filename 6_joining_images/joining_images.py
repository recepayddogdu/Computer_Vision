import cv2
import numpy as np

#resmi ice aktar
img = cv2.resize(cv2.imread("lenna.png"),(256,256))

cv2.imshow("original", img)

#horizontal (yatay) birleştirme
hor = np.hstack((img,img))
cv2.imshow("horizontal", hor)

#vertical (dikey) birleştirme
ver = np.vstack((img,img))
cv2.imshow("vertical", ver)

cv2.waitKey(0)
cv2.destroyAllWindows()