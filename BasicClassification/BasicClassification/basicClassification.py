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
#plt.figure()
#plt.imshow(train_images[0]) 
#plt.colorbar()
#plt.grid(False)
#plt.show()

# Skalierung der Trainingsimages von 0 bis 255 auf 0 bis 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Zeige eine Sammlung mit den Beispielbilder an
#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()

# 1. Schritt: build a model
# Ein neuronales Netz besteht aus mehreren Schichten "Layern". Die erste Schicht ist die Input-Schicht. Diese Schicht transformiert den 2D-Input in ein 1D Input.
# Aus einer 2x2 Matrix wird ein 1x4 Vektor: keras.layers.Flatten(input_shape=(28, 28))
# Die nächsten beiden Layer sind vollverknüpft.
#   1. Layer hat 128 Knotenpunkte
#   2. Layer hat 10 Knotenpunkte, die gleichzeitig als Outputwerte dienen.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 2. Schritt: 
# 2.1 Loss function: Die Loss function misst wie das Model sich selbst traniert. 
# 2.2 Optimizer: Der Optimizer updatet das Model anhand der Ergebnisse der Loss function.
# 2.3 Metrics: Es dient zur Visualisierung der Trainings- und Testergebnisse.

model.compile(optimizer = 'adam', 
              loss      = 'sparse_categorical_crossentropy',
              metrics   = ['accuracy'])

# 3. Schritt: Starte das Training
model.fit(train_images, train_labels, epochs=5)

# 3.1 Schritt: Speichere die Gewichte
# model.save_weights('./models/predictBasicCalssification')
# 3.1.1 Lade die Gewichte
# model = create_model()
# model.load_weights('./checkpoints/my_checkpoint')
# loss,acc = model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# 3.2 Schritt: Speichere das gesamte Model
model.save('./models/predictBasicCalssification.h5')
# 3.2.1 Lade das Model
#new_model = keras.models.load_model('my_model.h5')
#new_model.summary()

# 3.3

# 4. Schritt: Bertung der Genauigkeit
# 4.1 Das Model wird gegen das TestDatenSet getestet
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
# Ergebnis: Die Vorhersage ist ein wenig schlechter wie das Ergebnis gegenüber der Trainingsmenge. Dies nennt man auch "Overfitting".

# 5. Schritt: Mache eine Vorhersage
predictions = model.predict(test_images)
predictions[0]
np.argmax(predictions[0])


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# 6. Schritt: Visualisiere die Ergebnisse in Kombination mit den Bilder
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

