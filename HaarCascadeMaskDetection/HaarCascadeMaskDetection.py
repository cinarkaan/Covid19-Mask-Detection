from __future__ import print_function
import cv2 as cv
import argparse

# Denetleme Sonucu Bilgilerini Ekranda Yaz
def getinfo(faces,frame):
    font = cv.FONT_HERSHEY_TRIPLEX
    if len(faces) == 0:
        cv.putText(frame,'Lutfen maske takiniz',(20,40),font,0.8,(0,0,0),2)
        print("Maske Takılmamaıştır")
    else:
        cv.putText(frame,'Maske Algilandi',(20,40),font,0.8,(0,255,51))
        print("Maske takılmıştır")

def detecAndDisplay(frame):
    frame_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY) #Goruntuyu SiyahBeyaz Yap
    frame_gray = cv.equalizeHist(frame_gray) # Histogram eşitle

    #Yuz Algılama
    faces = face_cascade.detectMultiScale(frame_gray) # Algılamayı Baslat
    print(faces) # Algılanan Karenin Kordinatları
    getinfo(faces,frame) # Bilgileri Yaz
    for(x,y,w,h) in faces: 
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Algılanan Kareyi Diktörtgen İçine Al
    cv.imshow('Capture - Face detection', frame) # Ekranda Göster

# Maske ve Kamera Bilgileri Tanımlanıyor
parser = argparse.ArgumentParser(description='Cascade sınıflandırıcı')
parser.add_argument('--face_cascade',help='Sınıflandirici yolu',default='C:\\Users\\\Kaan\\\Documents\\dataset\\maskdetection.xml')
parser.add_argument('--camera',help='Kamera aygıtı secimi', type=int, default=0)
args = parser.parse_args()

face_cascade_name = args.face_cascade

print(face_cascade_name) # Maske dedektorü yolu 

print(cv.cuda.getCudaEnabledDeviceCount()) # Aktif Cuda Aygiti Sayisi
 
face_cascade = cv.CascadeClassifier(face_cascade_name) # Denetimi Yapacak Classifier Sınıfı 

if face_cascade.empty(): # Maske Algılayıcı Yuklemesi Basarilimi
    print('--(!) Yuz tanimlayicisi basarili bir sekilde yuklenemedi')
    exit(0)

camera_device = args.camera

cap = cv.VideoCapture(camera_device) # Web Camera Aygıtını Kullan

if not cap.isOpened: # Kamera Aygıtı Yukleme Basarilimi 
    print('--(!) Kamera aygitina erisirken hata olustu')
    exit(0)

while True:
    ret, frame = cap.read() # Web Camdan Bilgi Oku
    if frame is None:
        print("--(!)Kare Algılanmadı--")
        break
    
    detecAndDisplay(frame) # Denetle 

    if cv.waitKey(10) == 27: # Escape ile Cikis Yap
        break
