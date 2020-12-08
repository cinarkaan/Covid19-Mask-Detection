import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras import layers

data=np.load('data.npy')
target=np.load('target.npy')

maskmodel = Sequential()

maskmodel.add(layers.Conv2D(32,(3,3),input_shape=data.shape[1:]))
maskmodel.add(layers.MaxPooling2D(pool_size=(2,2)))

maskmodel.add(layers.Conv2D(64,(3,3)))
maskmodel.add(layers.Activation('relu'))
maskmodel.add(layers.MaxPooling2D(pool_size=(2,2)))

maskmodel.add(layers.Conv2D(128,(3,3)))
maskmodel.add(layers.Activation('relu'))
maskmodel.add(layers.MaxPooling2D(pool_size=(2,2)))

maskmodel.add(layers.Conv2D(256,(3,3)))
maskmodel.add(layers.Activation('relu'))
maskmodel.add(layers.MaxPooling2D(pool_size=(2,2)))

maskmodel.add(layers.Flatten())
maskmodel.add(layers.Dropout(0.5))
maskmodel.add(layers.Dense(64,activation='relu'))
maskmodel.add(layers.Dropout(0.4))

maskmodel.add(layers.Dense(2,activation='softmax'))

adam = tf.keras.optimizers.Adam(0.001)

maskmodel.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

maskmodel.summary()

train_history = maskmodel.fit(data,target,batch_size=100,epochs=50,validation_split=0.2,verbose=2,shuffle=True)

plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.legend(['accuracy','val_accuracy'])

maskmodel.save('maskmodel.h5')

