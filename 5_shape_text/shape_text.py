import cv2
import numpy as np

# siyah resim olustur
img = np.zeros((512, 512, 3), np.uint8)
print(img.shape)

#cizgi ekleme
cv2.line(img,       #resim
         (0,0),     #baslangic noktasi
         (512,512), #bitis noktasi
         (0,255,0), #renk
         3, )       #kalinlik

#dikdortgen ekleme
cv2.rectangle(img,       #resim
         (0,0),          #baslangic noktasi
         (256,256),      #bitis noktasi
         (255,200,0),    #renk
         cv2.FILLED)     #doldurma       
         


#cember cizimi
cv2.circle(img,         #resim
           (300,300),   #merkez
           45,          #yaricap
           (0,0,255),   #renk
           cv2.FILLED   #doldurma
           )

#Metin ekleme
cv2.putText(img,                       #resim
            "TEXT",                    #text
            (0,511),                   #baslangic noktasi
            cv2.FONT_HERSHEY_COMPLEX,  #font
            1.5,                       #kalinlik
            (255,255,255))             #renk

cv2.imshow("text", img)

if cv2.waitKey(0):
    cv2.destroyAllWindows()
    
