# Computer Vision Notes

Görüntülerin koordinat ekseninde **x** ve **y** konumları aşağıdaki gibi olur;

![Untitled](images/Untitled.png)

## Resmi İçe Aktarma

```python
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
```

## Video İçe Aktar

```python
import cv2
import time

video_name = "MOT17-04-DPM.mp4"

# Video ice aktar
cap = cv2.VideoCapture(video_name)

# Video kontrol
if cap.isOpened() == False:
    print("Path Error")

# Videonun genislik ve yuksekligi
print("Width: ", cap.get(3))
print("Height: ", cap.get(4))
```

### **Video Okuma**

`cap.read(path)` fonksiyonu ile video okunur ve bu fonksiyon 2 değer döndürür. `ret` ve `frame` .

`ret` video okuma işleminin başarılı olduğu durumda `true`, başarısız olduğu durumda ise `false` değeri alır.

`frame` ise video okuma işlemi başarılı olduğunda videonun içerisinde bulunan her bir frame'i alır.

```python
# Video okuma
while True:
    ret, frame = cap.read()
    
    if ret == True:
        #uyari: goruntulerin yavas akmasi icin
        time.sleep(0.01) 
        
        cv2.imshow("video", frame)
    else: break

    # "q" tusuna basildiginda döngüden cik
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# videoyu serbest birak
cap.release()

#tum acik pencereleri kapat
cv2.destroyAllWindows()
```

## Kamera Açma ve Video Kaydı

```python
import cv2

# capture
cap = cv2.VideoCapture(0)

# Videonun frame genisligi ve yuksekligini al
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width,  height)

# video kaydet
# cv2.VideoWriter_fourcc(*"DIVX")
# frame'leri sikistirmak icin kullanilan codec kodu
# fps = 20
writer = cv2.VideoWriter("video_kaydi.mp4", 
                         cv2.VideoWriter_fourcc(*"DIVX"),
                         20, (width, height))
while True:
    ret, frame = cap.read()
    cv2.imshow("video", frame)
    
    # save
    writer.write(frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
writer.release()
cv2.destroyAllWindows()
```

## Yeniden Boyutlandır ve Kırp

Görüntü işleme projelerinde 1080x1080 boyutlu bir görüntü yerine 480x480 boyutlu bir görüntü ile çalışmak çok daha performanslı ve kolay olacaktır. Bu nedenle çalıştığımız verileri yeniden boyutlandırma veya kırpmaya ihtiyaç duyarız.

### **Yeniden Boyutlandırma**

```python
import cv2

# resmi oku
img = cv2.imread("lenna.png", 0)
print("Image size: ", img.shape)

cv2.imshow("Orginal", img)
if cv2.waitKey(0) & 0xFF == ord("q"): 
    cv2.destroyAllWindows()
```

lenna.png'nin siyah beyaz boyutuna baktığımızda `Image size: (512, 512)` çıktısı alırız.

![Untitled](images/Untitled%201.png)

Renkli boyutuna baktığımızda ise `Image size: (512, 512, 3)` çıktısı elde ederiz. Buradaki 3 boyutu, 3 farklı renk olduğu için oluşur. Red, Green ve Blue.

![Untitled](images/Untitled%202.png)

Görüntünün boyutunu (800, 800) boyutuna büyütelim;

```python
# yeniden boyutlandir
imgResized = cv2.resize(img, (800,800))
print("Resized image shape: ", imgResized.shape)

cv2.imshow("Resized Image", imgResized)
```

![Untitled](images/Untitled%203.png)

### **Kırpma**

y ekseninde 200. , x ekseninde 300. piksele kadar görüntüyü kırpalım;

```python
# kırp
imgCropped = img[:200, 0:300]

cv2.imshow("Cropped Image", imgCropped)
```

![Untitled](images/Untitled%204.png)

## Şekiller ve Metin

Bir nesne tespiti yaptığımız zaman görüntü üzerine kutucuk çizmemiz, yazı yazmamız gerekebiliyor. Bu nedenle görüntü üzerine şekil ve metin eklemeye ihtiyaç duyuluyor.

### Ç**izgi Ekleme**

```python
import cv2
import numpy as np

# siyah resim olustur
img = np.zeros((512, 512, 3), np.uint8)
print(img.shape)

#line ekleme
cv2.line(img,       #resim
         (0,0),     #baslangic noktasi
         (512,512), #bitis noktasi
         (0,255,0), #renk
         3, )       #kalinlik
         
cv2.imshow("line", img)

if cv2.waitKey(0):
    cv2.destroyAllWindows()
```

![Untitled](images/Untitled%205.png)

### **Dikdörtgen Ekleme**

```python
#dikdortgen ekleme
cv2.rectangle(img,       #resim
         (0,0),          #baslangic noktasi
         (256,256),      #bitis noktasi
         (255,200,0),    #renk
         3, )            #kalinlik
```

![Untitled](images/Untitled%206.png)

### **Dikdörtgen Doldurma**

```python
#dikdortgen ekleme
cv2.rectangle(img,       #resim
         (0,0),          #baslangic noktasi
         (256,256),      #bitis noktasi
         (255,200,0),    #renk
         cv2.FILLED)     #doldurma
```

![Untitled](images/Untitled%207.png)

### **Çember Çizimi**

```python
#cember cizimi
cv2.circle(img,         #resim
           (300,300),   #merkez
           45,          #yaricap
           (0,0,255),   #renk
           )

cv2.imshow("circle", img)
```

![Untitled](images/Untitled%208.png)

### **Çember Doldurma**

```python
#cember cizimi
cv2.circle(img,         #resim
           (300,300),   #merkez
           45,          #yaricap
           (0,0,255),   #renk
           cv2.FILLED   #doldurma
           )
```

![Untitled](images/Untitled%209.png)

### **Metin Ekleme**

```python
#Metin ekleme
cv2.putText(img,                       #resim
            "TEXT",                    #text
            (350,350),                 #baslangic noktasi
            cv2.FONT_HERSHEY_COMPLEX,  #font
            1.5,                       #kalinlik
            (255,255,255))             #renk

cv2.imshow("text", img)
```

![Untitled](images/Untitled%2010.png)

## Görüntü Birleştirme

```python
import cv2
import numpy as np

#resmi ice aktar
img = cv2.resize(cv2.imread("lenna.png"),(256,256))

cv2.imshow("original", img)

#horizontal (yatay) birleştirme
hor = np.hstack((img,img))
cv2.imshow("horizontal", hor)
```

![Untitled](images/Untitled%2011.png)

```python
#vertical (dikey) birleştirme
ver = np.vstack((img,img))
cv2.imshow("vertical", ver)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

![Untitled](images/Untitled%2012.png)

## Perspektif Çarpıtma (Warp Perspective)

Point1 ve Point2 belirlememiz gerekiyor. Point1'de yamuk resmimizin köşelerinin piksel değerleri yer alacak. Görüntüyü Paint ile açtığımızda sol alt köşede imlecin yer aldığı pikselin değerleri gözükmekte.

![Untitled](images/Untitled%2013.png)

Sol üst köşeden başlayıp  köşeleri dolaştığımızda sırasıyla köşe noktaların koordinatları;

- (204,3) - sol üst
- (2,474) - sol alt
- (540,147) - sağ üst
- (340,617) - sağ alt

olarak belirlendi.

Point2 olarak ise dönüştürmek istediğimiz yeni köşe koordinatlarını belirlememiz gerekiyor.

`cv2.getPerspectiveTransform()` fonksiyonu ile iki noktayı verdikten sonra transform için gerekli olan matrix otomatik olarak oluşturulmuş olacak.

```python
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
```

![Untitled](images/Untitled%2014.png)

## Görüntüleri Karıştırmak (Blending)

İki görüntüyü alpha ve beta katsayıları ile birbirine karıştıralım;

![Untitled](images/Untitled%2015.png)

![Untitled](images/Untitled%2016.png)

```python
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("img1.JPG")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("img2.JPG")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

#birlestirmek icin shape'ler aynı boyutta olmak zorunda
print(img1.shape)
print(img2.shape)

img1 = cv2.resize(img1, (600,600))
print(img1.shape)

img2 = cv2.resize(img2, (600,600))
print(img2.shape)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

#karistirilmis resim = alpha*img1+beta*img2
blended=cv2.addWeighted(src1=img1, alpha=0.5,
                        src2=img2, beta=0.5,
                        gamma=0)
plt.figure()
plt.imshow(blended)
```

![Untitled](images/Untitled%2017.png)

## Görüntü Eşikleme

Görüntülerdeki siyaha yakın pikseller 0'a yakın, beyaza yakın pikseller ise 255'e yakın değerler alır. Bu değerler ile görüntü eşikleme yöntemi uygulanarak görüntüdeki detaylar yok edilebilir ve ana hat elde edilebilir.

Örneğin görüntüdeki 60 değerine threshold uygulandığından aşağıdaki gibi sonuç elde edilir;

```python
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("img1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# %matplotlib auto
plt.figure()
plt.imshow(img, cmap = "gray")
plt.axis("off")
plt.show()

#esikleme

_, thresh_img = cv2.threshold(img, thresh=60, maxval=255,
                              type=cv2.THRESH_BINARY)
#THRESH_BINARY_INV ile tam tersi yapilabilir
plt.figure()
plt.imshow(thresh_img, cmap="gray")
plt.axis("off")
plt.show()
```

![Untitled](images/Untitled%2018.png)

Görüntüde bir bütün olan alanlar var, örneğin dağ, ağaç vs gibi. Threshold uygulanırken bu nesnelerin bölünmesi istenmeyebilir. Ya tamamı kaldırılmak istenir ya da tamamı dursun istenir. Bunun çözümü `adaptive threshold` yöntemidir.

```python
#Adaptive Threshold
thresh_img2 = cv2.adaptiveThreshold(img, 255, 
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY,
                                       11, 8)

plt.figure()
plt.imshow(thresh_img2, cmap="gray")
plt.axis("off")
plt.show()
```

![Untitled](images/Untitled%2019.png)