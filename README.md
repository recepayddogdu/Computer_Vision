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