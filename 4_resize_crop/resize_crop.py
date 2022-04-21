import cv2

# resmi oku
img = cv2.imread("lenna.png")
print("Image size: ", img.shape)

cv2.imshow("Original", img)

# yeniden boyutlandir
imgResized = cv2.resize(img, (800,800))
print("Resized image shape: ", imgResized.shape)

# cv2.imshow("Resized Image", imgResized)

# kırp
imgCropped = img[:200, 0:300]

cv2.imshow("Cropped Image", imgCropped)

if cv2.waitKey(0) & 0xFF == ord("q"): 
    cv2.destroyAllWindows()