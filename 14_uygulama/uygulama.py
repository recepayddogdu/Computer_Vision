# opencv kütüphanesini içe aktaralım
import cv2

# matplotlib kütüphanesini içe aktaralım
import matplotlib.pyplot as plt
import numpy as np

# resmi siyah beyaz olarak içe aktaralım
img = cv2.imread("14_uygulama/uygulama.jpg", 0)
color_img = cv2.cvtColor(cv2.imread("14_uygulama/uygulama.jpg"), cv2.COLOR_BGR2RGB)

# resmi çizdirelim
def imshow_img(img, title):
    plt.figure(), plt.imshow(img, cmap="gray"), plt.title(title)

imshow_img(img, "original")
imshow_img(color_img, "color img")

# resmin boyutuna bakalım
print(img.shape)

# resmi 4/5 oranında yeniden boyutlandıralım ve resmi çizdirelim
resized_img = cv2.resize(img, (img.shape[1]*4//5, img.shape[0]*4//5))
print(resized_img.shape)
#imshow_img(resized_img, "resized image")

# orijinal resme bir yazı ekleyelim mesela "kopek" ve resmi çizdirelim
cv2.putText(img, "Kopek",(403,80),cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0))
imshow_img(img, "original wText")

# orijinal resmin 50 threshold değeri üzerindekileri beyaz yap altındakileri siyah yapalım, 
# binary threshold yöntemi kullanalım ve resmi çizdirelim
_, binary_thresh_img = cv2.threshold(img, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
imshow_img(binary_thresh_img, "binary_thresh_img")
adaptive_thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
imshow_img(adaptive_thresh_img, "adaptive_thresh_img")

# orijinal resme gaussian bulanıklaştırma uygulayalım ve resmi çizdirelim
img_gb = cv2.GaussianBlur(img, ksize=(9,9), sigmaX=7)
imshow_img(img_gb, "gaussian blur")

#Dilate
kernel = np.ones((5,5), dtype=np.uint8)
dilate_img = cv2.dilate(img, kernel, iterations=1)
imshow_img(dilate_img, "dilate_img")

#Erode
kernel = np.ones((5,5), dtype=np.uint8)
erode_img = cv2.erode(img, kernel, iterations=1)
imshow_img(erode_img, "erode")

#Morphological Gradient
gradient = cv2.morphologyEx(img.astype(np.float32), cv2.MORPH_GRADIENT, kernel)
imshow_img(gradient, "morphological gradient")


# orijinal resme Laplacian  gradyan uygulayalım ve resmi çizdirelim
laplacian = cv2.Laplacian(img, ddepth=cv2.CV_64F)
cv2.imshow("laplacian",laplacian)
if cv2.waitKey(0):
    cv2.destroyAllWindows()

# orijinal resmin histogramını çizdirelim
img_hist = cv2.calcHist([color_img], channels=[1], mask=None, histSize=[160], ranges=[0,160])
plt.figure(), plt.plot(img_hist), plt.title("histogram")

plt.show()