import os
import socket
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('C:\\Users\\Kaan\\Bitirme\\MaskeTespit\\TensorflowMaskDetection\\maskmodel.h5')
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
sock.bind(('127.0.0.1',5002))
os.system('cls')
print("Server was started and it wait data streaming......")

while True:
  length, addr = sock.recvfrom(16)
  stringData,addr = sock.recvfrom(int(length))
  data = np.fromstring(stringData, dtype='uint8')
  recvimg=cv2.imdecode(data,cv2.IMREAD_GRAYSCALE)
  normalized=recvimg/255.0
  reshaped=np.reshape(normalized,(1,50,50,1))
  result=model.predict(reshaped)
  label=np.argmax(result,axis=1)[0]
  sock.sendto(str(label).encode(),addr)