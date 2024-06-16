from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D
from keras.layers import Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Charger les données
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Normalisation des données
mean = np.array([0.491, 0.482, 0.447])
std = np.array([0.202, 0.199, 0.201])
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Convertir les labels en catégories
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

# Création du modèle
model = Sequential([
    Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3)),  # conv1
    MaxPooling2D(pool_size=(2, 2)),  # pool1
    Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),  # conv2
    MaxPooling2D(pool_size=(2, 2)),  # pool2
    Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),  # conv3
    Flatten(),  # Aplatit le tenseur de sortie
    Dense(1000, activation='relu'),  # fc4
    Dense(10, activation='softmax')  # fc5
])

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# Entraînement du modèle
model.fit(X_train, Y_train, batch_size=100, epochs=5, validation_split=0.1, verbose=1)

# Évaluation du modèle
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test accuracy:', score[1])

# Importation de la fonction confusion_matrix
# Récupération des prédictions du modèle sur les données de test
predictions = model.predict(X_test)

# Récupération des labels réels des données de test
true_labels = np.argmax(Y_test, axis=1)

# Création d'une figure avec deux sous-graphiques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Affichage des 10 premières prédictions du modèle
ax1.bar(range(10), predictions[0])
ax1.set_title("10 premières prédictions du modèle")
ax1.set_xlabel("Classes")
ax1.set_ylabel("Probabilités")

# Affichage de la matrice de confusion
cm = confusion_matrix(true_labels, np.argmax(predictions, axis=1))
ax2.imshow(cm, cmap='Blues', interpolation='nearest')
ax2.set_title("Matrice de confusion")
ax2.set_xlabel("Classes réelles")
ax2.set_ylabel("Classes prédites")

# Affichage des valeurs de la matrice de confusion
thresh = cm.max() / 2
for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax2.text(j, i, cm[i][j], ha="center", va="center", color="white" if cm[i][j] > thresh else "black")

# Ajout des lignes de grille pour améliorer la lisibilité
ax2.set_xticks(np.arange(len(cm[0])))
ax2.set_yticks(np.arange(len(cm)))
ax2.set_xticklabels([str(x) for x in range(len(cm[0]))])
ax2.set_yticklabels([str(x) for x in range(len(cm))])
ax2.grid(True)

# Affichage de la figure
plt.show()