import keras
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

# Chargement du jeu de données CIFAR-10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Normalisation des valeurs de pixels entre 0 et 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Encodage à chaud des étiquettes
Y_train = keras.utils.to_categorical(Y_train, 10)
Y_test = keras.utils.to_categorical(Y_test, 10)

# Augmentation des données
data_generator = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=90,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

# Adapter le générateur de données aux données d'entraînement
data_generator.fit(X_train)

# Définir le modèle CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compiler le modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraîner le modèle avec l'augmentation des données
batch_size = 50
epochs = 3
model.fit_generator(data_generator.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs,
                    validation_data=(X_test, Y_test),
                    workers=4)

# Évaluer le modèle
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#visualisation pouyr comparer ce qui predit et ce qui est reel
perdiction=model.predict(X_test)
true_labels=np.argmax(Y_test,axis=1)

#visualiser les 10 premieres predictions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.bar(range(10), predictions[0])
ax1.set_title("10 premières prédictions du modèle")
ax1.set_xlabel("Classes")
ax1.set_ylabel("Probabilités")

#visualiser la matrice de confusion
cm=confusion_matrix(true_labels,np.argmax(predictions,axis=1))
ax2=confusion_matrix(true_labels,np.argmax(predictions,axis=1))
plt.show()

#visualiser les images mal classées


