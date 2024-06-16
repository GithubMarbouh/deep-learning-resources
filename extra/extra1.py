#%%
# importer les packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D
from tensorflow.keras.datasets import cifar10
#%%
# charger les données
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#%%
# afficher les données
x_train[0]
#%%
x_train=x_train/255.0
x_test=x_test/255.0
#%%
# afficher les données
x_train[0]
#%%
# creer le modele
model = Sequential()
#%%
# creer le resau de convolution(CNN)
model.add(Conv2D(256, (3, 3), activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
#%%
model.add(Flatten())
#%%
# Le completement connecté
model.add(Dense(64))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
#%%
model.summary()
#%%
# entrainer le modele
model.fit(x_train, y_train, epochs=10)
#%%
# evaluer le modele
test_loss, test_acc = model.evaluate(x_test, y_test)