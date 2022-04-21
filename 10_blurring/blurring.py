import cv2
import matplotlib.pyplot as plt
import numpy as np

# blurring(detayi azaltmak icin)
img = cv2.cvtColor(cv2.imread("10_blurring/NYC.jpg"), cv2.COLOR_BGR2RGB)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(),plt.imshow(img),plt.axis("off")
plt.title("original")

# Ortalama bulaniklastirma ornegi
dst2 = cv2.blur(img, ksize=(3,3))
plt.figure(), plt.imshow(dst2), plt.axis("off")
plt.title("Mean Blurring")

# Gaussian blur ornegi
gb = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=7)
plt.figure(), plt.imshow(gb), plt.axis("off")
plt.title("Gaussian Blur")

# Medyan blur ornegi
mb = cv2.medianBlur(img, ksize=3)
plt.figure(), plt.imshow(mb), plt.axis("off")
plt.title("Median Blur")

#Noise olusturma

def gaussianNoise(image):
    row, col, ch = image.shape
    mean = 0 # 0 ortalamali noise
    var = 0.05
    sigma = var**0.5

    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss

    return noisy

#Noise eklenebilmesi icin goruntunun 0-1 arasi normalize edilmesi gerekir
img = cv2.cvtColor(cv2.imread("10_blurring/NYC.jpg"),
                    cv2.COLOR_BGR2RGB)/255 #normalize edilmis goruntu


plt.figure(), plt.imshow(img), plt.axis("off")
plt.title("Normalized Image")

gaussianNoisyImage = gaussianNoise(img)
plt.figure(), plt.imshow(gaussianNoisyImage), plt.axis("off")
plt.title("gaussianNoisyImage")

#gauss blur
gb2 = cv2.GaussianBlur(gaussianNoisyImage, ksize=(3,3), sigmaX=7)
plt.figure(), plt.imshow(gb2), plt.axis("off")
plt.title("with Gaussian Blur")


# salt pepper noise
def saltPepperNoise(image):

    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.004 #beyaz nokta sayisini belirler

    noisy = np.copy(image)

    # salt
    num_salt = np.ceil(amount*image.size*s_vs_p)
    coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
    noisy[coords] = 1

    # pepper
    num_pepper = np.ceil(amount*image.size*(1-s_vs_p))
    coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
    noisy[coords] = 0

    return noisy

spImage = saltPepperNoise(img)
plt.figure(), plt.imshow(spImage), plt.axis("off")
plt.title("SaltPepper Noise")

mb2 = cv2.medianBlur(spImage.astype(np.float32), ksize=3)
plt.figure(), plt.imshow(mb2), plt.axis("off")
plt.title("with median Blur"), plt.show()