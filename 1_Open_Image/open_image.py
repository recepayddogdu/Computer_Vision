import cv2

#ice aktarma
img = cv2.imread("1_Open_Image\messi5.jpg", 0)

#Gorsellestirme
cv2.imshow("ilk resim", img)
k = cv2.waitKey(0) &0xFF #klavyeden tus al

if k == 27: #esc ise
    cv2.destroyAllWindows() #pencereleri kapat
elif k == ord("s"): #s ise
    cv2.imwrite("messi_gray.png", img) #kaydet
    cv2.destroyAllWindows() #pencereleri kapat