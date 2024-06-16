# Pour la reproductibilité
from numpy.random import seed
seed(1)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D
from keras.layers import Conv2D, Flatten
from tensorflow.random import set_seed
set_seed(2)
from tensorflow.keras.utils import to_categorical
# preparer les données
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# les donnes sont en 2D
print("X_train[0] avant : ", X_train[0])
print("X_test[0] avant : ", X_test[0])
print("Y_train[0] avant : ", Y_train[0])
print("Y_test[0] avant : ", Y_test[0])

# Reshape les donnes en 3D
X_train = X_train.reshape(60000, 28, 28, 1)
print("X_train[0] apres : ", X_train[0])
X_test = X_test.reshape(10000, 28, 28, 1)
print("X_test[0] apres : ", X_test[0])
Y_train = to_categorical(Y_train, 10)
print("Y_train[0] apres : ",Y_train[0])
Y_test = to_categorical(Y_test, 10)
print("Y_test[0] apres : ",Y_test[0])

# Création du modèle
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1)),  # Premier calque convolutif
    Activation('relu'),
    Conv2D(filters=32, kernel_size=(3, 3)),  # Deuxième calque convolutif
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),  # Calque de mise en commun maximale
    Flatten(),  # Aplatit le tenseur de sortie
    Dense(64),  # Calque caché entièrement connecté
    Activation('relu'),
    Dense(10),  # Calque de sortie
    Activation('softmax')
])
# Résumé du modèle
print(model.summary())
# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# Entraînement du modèle
model.fit(X_train, Y_train, batch_size=100, epochs=5, validation_split=0.1, verbose=1)
# Évaluation du modèle
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test accuracy:', score[1])