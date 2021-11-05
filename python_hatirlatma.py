# %% spyder tanıtımı
print("Hello World")

# %% değişkenler

tamsayi_degisken = 10
ondalikli_sayi = 12.3
print(tamsayi_degisken)

# 4 işlem özellikleri
pi_sayisi = 3.14
katsayi = 2

toplam = pi_sayisi + 1 
fark = pi_sayisi - 1
carpma = pi_sayisi*katsayi
bolme = pi_sayisi/katsayi

# print 
print("Toplam: ", toplam)
print("Toplam: {} ve fark: {}".format(toplam, fark))
print("Çarpma: %.1f, bölme: %.4f" % (carpma, bolme))

# değişkenler arası dönüşüm

carpma_int = int(carpma)
print(carpma_int)

tamsayi_float = float(tamsayi_degisken)
print(tamsayi_float)

# string: karakter dizileri
string = "merhaba dünya"
print(string)

resim_yolu = "veri" + "\\" + "img" + ".png" 
print(resim_yolu)

# %% python temel sözdizimi

# büyük ve küçük harf
temel = 6
TEMEL = 7

# yorum
"""
bu bölümde sözdizimi
    - büyük küçük harf
    - yorum
    - girinti
    - anahtar kelimer
"""

# girinti
if 5 < 10:    
    print("yes")
else:
    print("no")
    
# anahtar kelime
de = 4    
# def = 4

# sayilı degiskeni
sayi1 = 5
sayi2 = 2

# 1sayi = 7

# %% liste
"""
- bileşik veri türüdür ve çok yönlüdür
- [ 1, "a", 1.0]
- farklı veri tiplerini içerisinde barındırabilir
"""

liste = [1,2,3,4,5,6]
print(type(liste))

hafta = ["pazartesi", "salı", "çarşamba", "perşembe", "cuma", "cumartesi", "pazar"]
# ilk eleman
print(hafta[0])

# son eleman
print(hafta[6])
print(len(hafta)) # eleman sayisi
print(hafta[len(hafta)-1])
print(hafta[-1])

# liste 2-3-4: 1,2,3 indeks 
print(hafta[1:4]) # 1'den 4'e kadar 1 dahil - 4 dahil değil

# sayı listesi
sayi_listesi = [1,2,3,4,5,6]
sayi_listesi.append(7)
print(sayi_listesi)

# listeden eleman çıkarma
sayi_listesi.remove(4)
print(sayi_listesi)

# listeyi ters çevir
sayi_listesi.reverse()
print(sayi_listesi)

# listeyi sırala
sayi_listesi = [1,3,2,4,5,67,0]
sayi_listesi.sort()
print(sayi_listesi)

# %% tuple
"""
değiştirilemez ve sıralı bir veri tipidir.
(1,2,3)
"""
tuple_veritipi = (1,2,3,3,4,5,6)
# ilk elemanı
print(tuple_veritipi[0])

# 2. indeksten sonraki elemanları yazdır
print(tuple_veritipi[2:])

# count eleman 
print(tuple_veritipi.count(3))

tuple_xyz = (1,2,3)
x, y, z = tuple_xyz
print(x,y,z)

# %% deque

from collections import deque
dq = deque(maxlen = 3)

dq.append(1) # 1 ekle sonuna [1]
print(dq)

dq.append(2) # 2 ekle sonuna [1,2]
print(dq)

dq.append(3) # 3 ekle sonuna [1,2,3]
print(dq)

dq.append(4) # 4 ekle sonuna [2,3,4]
print(dq)

dq = deque(maxlen = 3)
dq.append(1) # 1 ekle sonuna [1]
print(dq)
dq.append(2) # 2 ekle sonuna [1,2]
print(dq)

dq.appendleft(3) # 3 ekle başına [3,1,2]
print(dq)

dq.clear()
print(dq)

# %% sözlük
"""
- bir çeşit karma tablo türüdür
- anahtar ve değer çiftlerinden oluşur
- { "anahtar": değer }
"""

dictionary = { "İstanbul" : 34,
               "İzmir"    : 35,
               "Konya"    : 42}
print(dictionary)


# istanbul anahtarının değeri
print(dictionary["İstanbul"])

# anahtarlar
print(dictionary.keys())

# değerler
print(dictionary.values())

# %% koşullu ifadeler if else statement
"""
bir bool ifadesine göre doğru yada yanlış olarak değerlendirilmesine
bağlı olarak farklı hesaplamalar veya eylemler gerçekleştiren 
bir ifadedir.
"""

# büyük küçük sayi karşılaştırması
sayi1 = 12.0
sayi2 = 20.0

if sayi1 < sayi2:
    print("sayi 1 küçüktür sayi2")
elif sayi1 > sayi2:
    print("sayi 1 büyüktür sayi2")
else:
    print("sayi 1 eşittir sayi2")


liste = [1,2,3,4,5]
deger = 1

if deger in liste:
    print("{} değeri listenin içerisindedir".format(deger))
else: 
    print("{} değeri listenin içerisinde değildir".format(deger))
    
    
dictionary = {"Türkiye" : "Ankara", "İngiltere" : "Londra", "İspanya": "Madrid"}

keys = dictionary.keys()
deger = "Türkiye"
if deger in keys:
    print("Evet")
else: print("Hayır")
    

bool1 = True
bool2 = True

if bool1 and bool2: print("Doğru")
else: print("Yanlış")

# %% döngüler
"""
- bir dizi üzerinde yineleme yapmak için kullanılan yapılardır.
- diziler: liste, tuple, string, sözlük, numpy pandas veri tipleri
"""

# for 
for i in range(1,11):
    print(i)
    
liste = [1,4,5,6,8,3,3,4,67]
print(sum(liste))

toplam = 0
for c in liste:
    toplam = toplam + c

print(toplam)
"""
0 = 0 + 1
1 = 1 + 4
5 = 5 + 6
11 = 11 + 8
toplam = 19 ...
"""

tup1 = ((1,2,3),(3,4,5))
for x, y, z in tup1:
    print(x,y,z)

# while
i = 0
while i < 4:
    print(i)
    i = i + 1

liste = [1,4,5,6,8,3,3,4,67]
sinir = len(liste)

her = 0
hesapla = 0
while her < sinir:
    hesapla = hesapla + liste[her]
    her = her + 1
print(hesapla)

# %% fonksiyonlar
"""
 - karmaşık işlemleri toplar vetek adımda yapmamızı sağlar
 - şablon
 - düzenleme
""" 

# kullanıcı tarafından tanımlanan fonk. 

def daireAlan(r):
    """
    Parameters
    ----------
    r : int - daire yarıçapı.

    Returns
    -------
    daire_alani : float - daire alanı
    """
    pi = 3.14
    daire_alani = pi*(r**2)

    # print(daire_alani)    
    
    return daire_alani

daireAlanıDegiskeni = daireAlan(3)

print(daireAlanıDegiskeni)

def daireCevre(r, pi = 3.14):
    """
    Parameters
    ----------
    r : int - daire yarıçapı.
    pi: float - pi sayısı

    Returns
    -------
    daire_cevre : float - daire çevresi
    """
    daire_cevre = 2*pi*r
    
    return daire_cevre
    
daireCevre = daireCevre(3)
print(daireCevre)


katsayi = 5

def katsayiCarpimi():
    global katsayi
    print(katsayi*katsayi)
    
katsayiCarpimi()
print(katsayi)


# boş fonk.
def bos():
    pass
    
# built in fonk.
liste = [1,2,3,4]
print(len(liste))
print(str(liste))
liste2 = liste.copy()
print(liste2)
print(max(liste2))
print(min(liste))

# lambda fonk.
"""
- ileri seviyeli
- küçük ve anonim bir işlemdir
"""

def carpma(x,y,z):
    return x*y*z

sonuc = carpma(2,3,4)
print(sonuc)

# aynı işlem with lambda
fonksiyon_lambda = lambda x,y,z : x*y*z
sonuc2 = fonksiyon_lambda(2,3,4)
print(sonuc2)

# %% yield
"""
- iterasyon - yineleme
- generator
- yield
"""

liste = [1,2,3]
for i in liste:
    print(i)
    
"""
generator yinelecileri
generator değerleri bellekte saklamaz yeri gelince anında üretirler
"""

generator = (x for x in range(1,4))
for i in generator:
    print(i)
    
"""
fonksiyon eğer return olarak generator döndürecek ise bunu return yerine yield anahtar sözcüğü ile yapar
"""   

def createGenerator():
    liste = range(1,4)
    for i in liste:
        yield i
        
generator = createGenerator()
print(generator)

for i in generator:
    print(i)

# %% numpy
"""
- matrisler için hesaplama kolaylığı sağlar
"""
import numpy as np

# 1*15 boyutunda bir array-dizi
dizi = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print(dizi)

print(dizi.shape) # array'in boyutu

dizi2 = dizi.reshape(3, 5)

print("Şekil: ",dizi2.shape)
print("Boyut: ",dizi2.ndim)
print("Veri tipi: ",dizi2.dtype.name)
print("Boy: ",dizi2.size) 

# array type
print("Type: ",type(dizi2))

# 2 boyutlu array
dizi2D = np.array([[1,2,3,4],[5,6,7,8],[9,8,7,5]])
print(dizi2D)

# sıfırlardan oluşan bir array
sifir_dizi = np.zeros((3,4))
print(sifir_dizi)

# birlerden oluşan bir array
bir_dizi = np.ones((3,4))
print(bir_dizi)

# bos array
bos_dizi = np.empty((3,4))
print(bos_dizi)

# arange(x,y,basamak)
dizi_aralik = np.arange(10,50,5)
print(dizi_aralik)

# linspace(x,y, basamak) 
dizi_bosluk = np.linspace(10,20,5)
print(dizi_bosluk)

# float array
float_array = np.float32([[1,2],[3,4]])
print(float_array)

# matematiksel işlemler
a = np.array([1,2,3])
b = np.array([4,5,6])

print(a+b)
print(a-b)
print(a**2)

# dizi elemanı toplama
print(np.sum(a))

# max değer
print(np.max(a))

# min değer
print(np.min(a))

# mean ortalama
print(np.mean(a))

# median ortalama
print(np.median(a))

# rastgele sayı üretme [0,1] arasında sürekli uniform 3*3
rastgele_dizi = np.random.random((3,3))
print(rastgele_dizi)

# indeks
dizi = np.array([1,2,3,4,5,6,7])
print(dizi[0])

# dizinin ilk 4 elemanı
print(dizi[0:4])

# dizinin tersi
print(dizi[::-1])

# 
dizi2D = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(dizi2D)

# dizinin 1. satır ve 1. sutununda bulunan elemanı
print(dizi2D[1,1])

# 1. sütun ve tüm satılar
print(dizi2D[:,1])

# satır 1, sütun 1,2,3
print(dizi2D[1,1:4])

# dizinin son satır tüm sütunları
print(dizi2D[-1, :])

# 
dizi2D = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(dizi2D)

# vektör haline getirme
vektor = dizi2D.ravel()
print(vektor)

maksimum_sayinin_indeksini = vektor.argmax()
print(maksimum_sayinin_indeksini)

# %% pandas
"""
- hızlı güçlü ve esnek
"""

import pandas as pd

# sözlük oluştur
dictionary = {"isim" : ["ali","veli","kenan","murat","ayse","hilal"],
              "yas"  : [15,16,17,33,45,66],
              "maas" : [100,150,240,350,110,220]}

veri = pd.DataFrame(dictionary)
print(veri)

# ilk 5 satır
print(veri.head())
print(veri.columns)
# veri bilgisi
print(veri.info())

# istatistiksel özellikler
print(veri.describe())

# yas sütunu
print(veri["yas"])

# sütun eklemek
veri["sehir"] = ["Ankara","İstanbul","Konya","İzmir","Bursa","Antalya"]
print(veri)

# yas sütunu
print(veri.loc[:,"yas"])

# yas sütunu ve 3 satır
print(veri.loc[:2,"yas"])

# yas ve şehir arası sütunu ve 3 satır
print(veri.loc[:2,"yas":"sehir"])

# yas ve şehir arası sütunu ve 3 satır
print(veri.loc[:2,["yas","isim"]])

# satırları tersten yazdır.
print(veri.loc[::-1,:])

# yas sütunu with iloc
print(veri.iloc[:,1])

# ilk 3 satır ve yaş ve isim
print(veri.iloc[:3,[0,1]])

# filtreleme
dictionary = {"isim" : ["ali","veli","kenan","murat","ayse","hilal"],
              "yas"  : [15,16,17,33,45,66],
              "sehir": ["İzmir","Ankara","Konya","Ankara","Ankara","Antalya"]}

veri = pd.DataFrame(dictionary)
print(veri)

# ilk olarak yaşa göre bir filtre yas > 22
filtre1 = veri.yas > 22
filtrelenmis_veri = veri[filtre1]
print(filtrelenmis_veri)

# ortalama yas 
ortalama_yas = veri.yas.mean()

veri["YAS_GRUBU"] = ["kucuk" if ortalama_yas > i else "buyuk" for i in veri.yas]
print(veri)

# birleştirme
sozluk1 = {"isim": ["ali","veli","kenan"],
              "yas" : [15,16,17],
              "sehir": ["İzmir","Ankara","Konya"]} 
veri1 = pd.DataFrame(sozluk1)

# veri seti 2 oluşturalım
sozluk2 = {"isim": ["murat","ayse","hilal"],
              "yas" : [33,45,66],
              "sehir": ["Ankara","Ankara","Antalya"]} 
veri2 = pd.DataFrame(sozluk2)

# dikey
veri_dikey = pd.concat([veri1, veri2], axis = 0)

# yatay
veri_yatay = pd.concat([veri1, veri2], axis = 1)


# %% matplotlib
"""
- görselleştirme
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4])
y = np.array([4,3,2,1])

plt.figure()
plt.plot(x,y, color="red",alpha = 0.7, label = "line")
plt.scatter(x,y,color = "blue",alpha= 0.4, label = "scatter")
plt.title("Matplotlib")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.xticks([0,1,2,3,4,5])
plt.legend()
plt.show()

fig, axes = plt.subplots(2,1, figsize=(9,7))
fig.subplots_adjust(hspace = 0.5)

x = [1,2,3,4,5,6,7,8,9,10]
y = [10,9,8,7,6,5,4,3,2,1]


axes[0].scatter(x,y)
axes[0].set_title("sub-1")
axes[0].set_ylabel("sub-1 y")
axes[0].set_xlabel("sub-1 x")

axes[1].scatter(y,x)
axes[1].set_title("sub-2")
axes[1].set_ylabel("sub-2 y")
axes[1].set_xlabel("sub-2 x")

# random resim
plt.figure()
img = np.random.random((50,50))
plt.imshow(img, cmap = "gray") # 0(siyah) 1(beyaz) -> 0.5(gri) 
plt.show()

# %% OS
import os

print(os.name)

currentDir = os.getcwd()
print(currentDir)

# new folder
folder_name = "new_folder"
os.mkdir(folder_name)

new_folder_name = "new_folder_2"
os.rename(folder_name, new_folder_name)

os.chdir(currentDir+"\\"+new_folder_name)
print(os.getcwd())

os.chdir(currentDir)
print(os.getcwd())

files = os.listdir()

for f in files:
    if f.endswith(".py"):
        print(f)

os.rmdir(new_folder_name)

for i in os.walk(currentDir):
    print(i)

os.path.exists("python_hatırlatma.py")