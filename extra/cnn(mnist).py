import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from importlib import reload
sys.path.append('..')
from tensorflow.keras.datasets import mnist


data = mnist.load_data()

# telechargement des donnees
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

# normalisation
print("Before normalization: Min={} ,max() ".format(x_train.min(), x_train.max()))
xmax = x_train.max()
x_train = x_train / xmax
x_test = x_test / xmax
print("After normalization: Min={} ,max() ".format(x_train.min(), x_train.max()))

data.plot_images(x_train, y_train, [27],x_size=5, y_size=5,colorbar=True,save_as='01-one-digit')
data.plot_images(x_train, y_train,range(5,41),columns=12,save_as='02-many-digits')

# Creation du modele
model = keras.Sequential()
model.add(keras.layers.Input(shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(8, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 512
epochs = 10
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(x_test, y_test),
                    verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss:, {score[0]:4.4f}')
print(f'Test accuracy:, {score[1]:4.4f}')

#Plot results
y_sigmoid = model.predict(x_test)
y_pred = np.argmax(y_sigmoid, axis=1)
data.plot_images(x_test, y_test, range(0, 200), columns=12,x_size=1,y_size=1, y_pred=y_pred, save_as='03-prediction')

#erreur
errors = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]
errors= errors[:min(24, len(errors))]
data.plot_images(x_test, y_test, errors, columns=12, x_size=1, y_size=1, y_pred=y_pred, save_as='04-errors')

#Matrice de confusion
data.plot_confusion_matrix(y_test, y_pred, save_as='05-confusion-matrix')
data.end()






