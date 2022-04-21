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