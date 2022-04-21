import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("11_morphological/datai_team.jpg", 0)
plt.figure(), plt.imshow(img), plt.axis("off")
plt.title("original")

# Erozyon
kernel = np.ones((5,5), dtype=np.uint8)
# iterations: kac kez erozyon yapilacak
result = cv2.erode(img, kernel, iterations=1)

plt.figure(), plt.imshow(result), plt.axis("off")
plt.title("Erode")

#Dilation
result2 = cv2.dilate(img, kernel, iterations=1)
plt.figure(), plt.imshow(result2), plt.axis("off")
plt.title("Dilate")

# White Noise
whiteNoise = np.random.randint(0,2, #0-1 arasinda
                            img.shape[:2]) #ch dahil degil
whiteNoise = whiteNoise*255
plt.figure(), plt.imshow(whiteNoise, cmap="gray"), plt.axis("off")
plt.title("WhiteNoise")

noise_img = whiteNoise+img
plt.figure(), plt.imshow(noise_img, cmap="gray"), plt.axis("off")
plt.title("Noise Img")

#Açılma (Opening)
opening = cv2.morphologyEx(noise_img.astype(np.float32), cv2.MORPH_OPEN, kernel)

plt.figure(), plt.imshow(opening, cmap="gray"), plt.axis("off")
plt.title("Opening")

#Black Noise
blackNoise = np.random.randint(0,2, #0-1 arasinda
                            img.shape[:2]) #ch dahil degil
blackNoise = blackNoise*-255
plt.figure(), plt.imshow(blackNoise, cmap="gray"), plt.axis("off")
plt.title("BlackNoise")

blackNoise_img = blackNoise+img
blackNoise_img[blackNoise_img<=245]=0

plt.figure(), plt.imshow(blackNoise_img, cmap="gray"), plt.axis("off")
plt.title("BlackNoise Image")

#Kapatma (Closing)
closing = cv2.morphologyEx(noise_img.astype(np.float32), cv2.MORPH_CLOSE, kernel)

plt.figure(), plt.imshow(closing, cmap="gray"), plt.axis("off")
plt.title("Closing")

#Morphological Gradient
gradient = cv2.morphologyEx(img.astype(np.float32), cv2.MORPH_GRADIENT, kernel)

plt.figure(), plt.imshow(gradient, cmap="gray"), plt.axis("off")
plt.title("Gradient")
plt.show()