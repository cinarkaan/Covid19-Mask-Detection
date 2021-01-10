import os
import cv2
import socketserver
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('C:\\Users\\Kaan\\Bitirme\\MaskeTespit\\TensorflowMaskDetection\\maskmodel2.h5',None,True)
os.system('cls')
print("Server was started and it wait data streaming......")

class MyUDPHandler(socketserver.BaseRequestHandler):

    def handle(self):
        stringData = self.request[0].strip()
        data = np.fromstring(stringData, dtype='uint8')
        recvimg=cv2.imdecode(data,0)
        normalized=recvimg / 255.0
        reshaped=np.reshape(normalized,(1,50,50,1))
        result=model.predict(reshaped,None,1,None,None,10,1,True)
        label=np.argmax(0,25,axis=1)[0]
        self.request[1].sendto(str(label).encode(),self.client_address)
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    HOST,PORT = "127.0.0.1",5002
    with socketserver.UDPServer((HOST,PORT),MyUDPHandler) as server:
        server.serve_forever()