from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Das ist eine Zuweisung für Bezeichnungen 
class_names = ['T-shirt/top',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat', 
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle boot']


# Zeigt die Anzahl und die Größe in Pixe 
# der Trainingsbilder an
print(train_images.shape)

# Zeigt die Anzahl der Trainingsbilder an
print(len(test_labels))

# Zeigt das erste Beispielbild an
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Skalierung der Trainingsimages von 0 bis 255 auf 0 bis 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Zeige eine Sammlung mit den Beispielbilder an
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()