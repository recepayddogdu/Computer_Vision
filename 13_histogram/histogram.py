import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("13_histogram/red_blue.jpg")
img_vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(), plt.imshow(img_vis), plt.axis("off")
plt.title("original")

print(img.shape)

img_hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0,256])
print(img_hist.shape)
plt.figure(), plt.plot(img_hist)

color = ("b", "g", "r")
plt.figure()
for i, c in enumerate(color):
    hist = cv2.calcHist([img], channels=[i], mask=None, histSize=[256], ranges=[0,256])
    plt.plot(hist, color=c)

# Golden Gate Ornegi
golden_gate = cv2.imread("13_histogram/goldenGate.jpg")
golden_gate_vis = cv2.cvtColor(golden_gate, cv2.COLOR_BGR2RGB)

plt.figure(), plt.imshow(golden_gate_vis), plt.axis("off")
plt.title("GoldenGate")

print(golden_gate.shape)

#goruntu boyutunda maske olusturma
mask = np.zeros(golden_gate.shape[:2], np.uint8)


#belirli bir alan disinda tum pikselleri beyaz yap
mask[1500:2000, 1000:2000] = 255
plt.figure(), plt.imshow(mask, cmap="gray"), plt.title("mask")

masked_img_vis = cv2.bitwise_and(golden_gate_vis,golden_gate_vis,mask=mask)
plt.figure(), plt.imshow(masked_img_vis, cmap="gray"), plt.title("mask")

masked_img = cv2.bitwise_and(golden_gate,golden_gate,mask=mask)


#histogram olusturma
masked_img_hist_red = cv2.calcHist([masked_img], channels=[2], mask=mask, histSize=[256], ranges=[0,256])
plt.figure(),plt.title("masked_img_hist_red"), plt.plot(masked_img_hist_red)
masked_img_hist_green = cv2.calcHist([masked_img], channels=[1], mask=mask, histSize=[256], ranges=[0,256])
plt.figure(),plt.title("masked_img_hist_green"), plt.plot(masked_img_hist_green)
masked_img_hist_blue = cv2.calcHist([masked_img], channels=[0], mask=mask, histSize=[256], ranges=[0,256])
plt.figure(),plt.title("masked_img_hist_blue"), plt.plot(masked_img_hist_blue)

color = ("b", "g", "r")
plt.figure()
for i, c in enumerate(color):
    masked_img_hist = cv2.calcHist([masked_img], channels=[i], mask=mask, histSize=[256], ranges=[0,256])
    plt.plot(masked_img_hist, color=c)

#Histogram esitleme (equalization)
img = cv2.imread("13_histogram/hist_equ.jpg", 0)
plt.figure(), plt.imshow(img, cmap="gray")

img_hist=cv2.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.figure(),plt.title("img_hist"), plt.plot(img_hist)

eq_hist = cv2.equalizeHist(img)
plt.figure(), plt.imshow(eq_hist, cmap="gray")

eq_img_hist=cv2.calcHist([eq_hist], channels=[0], mask=None, histSize=[256], ranges=[0,256])
plt.figure(),plt.title("eq_img_hist"), plt.plot(eq_img_hist)


plt.show()