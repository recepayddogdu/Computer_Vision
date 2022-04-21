import cv2
import numpy as np

img = cv2.imread("kart.png")

cv2.imshow("original", img)

width = 400
height = 500

#cevirmek istenilen koseler
pts1 = np.float32([[204,3],[2,474],[540,147],[340,617]])

pts2 = np.float32([[0,0],[0, height],[width,0],[width,height]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)

# nihai donusturulmus resim
imgOutput = cv2.warpPerspective(img, matrix, (width,height))

cv2.imshow("nihai resim", imgOutput)

cv2.waitKey(0)
cv2.destroyAllWindows()