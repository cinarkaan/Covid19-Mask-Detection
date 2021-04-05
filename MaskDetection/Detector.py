import cv2
import os
import numpy as np
import tensorflow as tf
from imutils.video import FPS
import array as arr

#Gpu belleğinde belirtilen miktarda yer ayırt ve kullan
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)

class faceDetector:

    avarageFPS = None

    #Face Detector Class'ının yapılandırıcı metodu
    def __init__(self,use_cuda = False):
        #with tf.device('/cpu:0'): # cpu kullanarak yap
            #Burada framede işlenecek kısımlar için modeller yükleniyor
        self.model = tf.keras.models.load_model('maskmodel3')
        self.faceModel = cv2.dnn.readNetFromCaffe("face_detection_dnn\\res10_300x300_ssd_iter_140000.prototxt",
        caffeModel="face_detection_dnn\\res10_300x300_ssd_iter_140000.caffemodel")

        self.labels_dict={0:'MASK',1:'NO MASK'}
        self.color_dict={0:(0,255,0),1:(0,0,255)}
        self.mask_status = arr.array('i',[0,0])

        if use_cuda:
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    #Resim işlenmesi için gerekli metod
    def detectInImage(self,imgName):

        self.img = cv2.imread(imgName)
        (self.height, self.width) = self.img.shape[:2]

        self.detectInFrame()

        cv2.imshow("Output",self.img)
        cv2.waitKey(0)

    

    #Video işlenmesi için gerekli metod
    def detectInVideo(self, videoName):

        cap = cv2.VideoCapture(videoName)

        if cap.isOpened() == False:
            print ("Error opening video")
            return
        
        (success,self.img) = cap.read()
        (self.height, self.width) = self.img.shape[:2]

        fps = FPS().start()

        while success:
            self.detectInFrame()
            cv2.imshow("Output",self.img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            fps.update()
            (success, self.img) = cap.read()

        fps.stop()
        print("Elapsed time : {:.2f}".format(fps.elapsed()))
        print("FPS : {:.2f}".format(fps.fps()))

        self.avarageFPS = fps.fps()

        cap.release()
        cv2.destroyAllWindows()

    #Video işlenmesi için gerekli metod
    def detectInFrame(self):
        # Bu method
        #swapRB:Varsayılan görüntünün rgb mi bgr formatındamı olduğunu soran parametre
        #scaleFactor: Görüntünün 0 ile 1 arasında ölçeklenmesini ayarlayacağımız parametre
        #size:Sinir ağına vereceğimiz girdi boyutu
        #crop:False olarak ayarlandığında en boy oranı korunacak boyutuna göre yeniden boyutlandırılacaktır
        #işlemlerini yaparak görüntüyü ön işlemeye sokar
        blob = cv2.dnn.blobFromImage(self.img, 1.0, (300,300),(104.0,177.0,132.0),
        swapRB = False, crop = False)
        #CNN e giriş olarak görüntüyü veriyoruz
        self.faceModel.setInput(blob)
        #Tahmini yaptırıp geriye resimdeki yüzlerin koordinatlarını bize veriyor
        predictions = self.faceModel.forward()

        #Her bir koordinatlar ile normal resimin içindeki yüzleri teker teker kırpıp maske tespit metoduna iletiyoruz
        for i in range(0, predictions.shape[2]):
            if  predictions[0, 0, i, 2] > 0.5:    
                bbox = predictions[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                (xmin, ymin, xmax, ymax) = bbox.astype("int") 
                predict = self.maskDetector(self.img[ymin:ymax, xmin:xmax])
                if predict == -1:
                    cv2.rectangle(self.img, (xmin, ymin), (xmax, ymax),(255 , 0, 0), 2)
                else:            
                    cv2.rectangle(self.img, (xmin, ymin), (xmax, ymax),self.color_dict[predict], 2)
                    cv2.putText(self.img,self.labels_dict[predict],(xmin,ymin),cv2.FONT_HERSHEY_SIMPLEX,1,self.color_dict[predict],2,cv2.LINE_AA)
                    if (predict != -1):
                        self.mask_status[predict] += 1

   #Maske tespit işlemi yapılıyor
    def maskDetector(self,crop_img):
        if crop_img.shape[1] == 0 or crop_img.shape[0] == 0:
            return -1
        else:
            recvimg = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
            recvimg= cv2.resize(recvimg,(50,50))
            normalized= recvimg / 255.0
            reshaped= np.reshape(normalized,(1,50,50,1))
            #with tf.device('/cpu:0'):
            result= self.model.predict(reshaped)
            label= np.argmax(result,axis=1)[0]
            return label