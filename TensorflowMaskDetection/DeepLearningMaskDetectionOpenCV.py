import cv2
import numpy as np
from tensorflow.keras.models import load_model

CasacdePath = "C:\\Users\\Kaan\\Documents\\HaarCascadeMask\\data\\haarcascades\\haarcascade_frontalface_default.xml"
model = load_model('TensorflowMaskDetection\\maskmodel.h5')

faceCascade = cv2.CascadeClassifier(CasacdePath)
source = cv2.VideoCapture(0)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

#MODEL FARKLI RESİMLERLE TEST EDİLİYOR
#test_image = "C:\\Users\\Kaan\\Documents\\MaskData\\maskdatasettest\\maske.jpg" 
#img = cv2.imread(test_image)

while True:

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(50,50))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,50,50,1))
        result=model.predict(reshaped)
        label=np.argmax(result,axis=1)[0]
      

        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_TRIPLEX,0.8,color_dict[label],2)
        
        
    cv2.imshow('Mask Detection',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()