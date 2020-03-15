import sys
import os
import pandas as pd
import numpy as np
import pickle
import cv2
from os import listdir
from tqdm import tqdm
#from __future__ import unicode_literals
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import shuffle
import random
from keras.models import model_from_json

Pepper__bell___Bacterial_spot = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Pepper,_bell___Bacterial_spot"
Pepper__bell___healthy = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Pepper,_bell___healthy"
Potato___Early_blight = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Potato___Early_blight"
Potato___healthy = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Potato___healthy"
Potato___Late_blight = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Potato___Late_blight"
Tomato__Target_Spot = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Tomato___Target_Spot"
Tomato__Tomato_mosaic_virus = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Tomato___Tomato_mosaic_virus"
Tomato__Tomato_YellowLeaf__Curl_Virus = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Tomato___Tomato_Yellow_Leaf_Curl_Virus"
Tomato_Bacterial_spot = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Tomato___Bacterial_spot"
Tomato_Early_blight = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Tomato___Early_blight"
Tomato_healthy = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Tomato___healthy"
Tomato_Late_blight = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Tomato___Late_blight"
Tomato_Leaf_Mold = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Tomato___Leaf_Mold"
Tomato_Septoria_leaf_spot = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Tomato___Septoria_leaf_spot"
Tomato_Spider_mites_Two_spotted_spider_mite = "C:/Users/eduar/Escritorio/Eduardo-MCC/PlantVillage/PlantVillage/train/Tomato___Spider_mites Two-spotted_spider_mite"


ancho_Imagen = 256
largo_Imagen = 256
depth = 3
EPOCAS = 25
lr = 1e-3
batch_size= 32
x = []
x1 = []
y = []
y1 = []
datos_entrenamiento = []
temp = []
label_binarizer = LabelBinarizer()

for img in tqdm(os.listdir(Pepper__bell___Bacterial_spot)[:200]):
	label = 'Pepper__bell___Bacterial_spot'
	y.append(label)
	path = os.path.join(Pepper__bell___Bacterial_spot,img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256,256))
	x.append([np.asarray(img)])
print(np.array(x).shape)

for img in tqdm(os.listdir(Pepper__bell___healthy)[:200]):
	label = 'Pepper__bell___healthy'
	y.append(label)
	path = os.path.join(Pepper__bell___healthy,img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256,256))
	x.append([np.asarray(img)])
print(np.array(x).shape)

for img in tqdm(os.listdir(Potato___Early_blight)[:200]):
	label = 'Potato___Early_blight'
	y.append(label)
	path = os.path.join(Potato___Early_blight,img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256,256))
	x.append([np.asarray(img)])
print(np.array(x).shape)

for img in tqdm(os.listdir(Potato___healthy)[:200]):
	label = 'Potato___healthy'
	y.append(label)
	path = os.path.join(Potato___healthy,img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256,256))
	x.append([np.asarray(img)])
print(np.array(x).shape)

for img in tqdm(os.listdir(Tomato__Target_Spot)[:200]):
	label = 'Tomato__Target_Spot'
	path = os.path.join(Tomato__Target_Spot,img)
	img = cv2.imread(path)
	y.append(label)
	img = cv2.resize(img, (256,256))
	x.append([np.asarray(img)])
print(np.array(x).shape)

for img in tqdm(os.listdir(Tomato__Tomato_mosaic_virus)[:200]):
	label = 'Tomato__Tomato_mosaic_virus'
	y.append(label)
	path = os.path.join(Tomato__Tomato_mosaic_virus,img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256,256))
	y.append([np.asarray(img)])
print(np.array(x).shape)

for img in tqdm(os.listdir(Tomato__Tomato_YellowLeaf__Curl_Virus)[:200]):
	label = 'Tomato__Tomato_YellowLeaf__Curl_Virus'
	y.append(label)
	path = os.path.join(Tomato__Tomato_YellowLeaf__Curl_Virus,img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256,256))
	x.append([np.asarray(img)])
print(np.array(x).shape)

for img in tqdm(os.listdir(Tomato_Bacterial_spot)[:200]):
	label = 'Tomato_Bacterial_spot'
	y.append(label)
	path = os.path.join(Tomato_Bacterial_spot,img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256,256))
	x.append([np.asarray(img)])
print(np.array(x).shape)

for img in tqdm(os.listdir(Tomato_Early_blight)[:200]):
	label = 'Tomato_Early_blight'
	y.append(label)
	path = os.path.join(Tomato_Early_blight,img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256,256))
	x.append([np.asarray(img)])
print(np.array(x).shape)

for img in tqdm(os.listdir(Tomato_healthy)[:200]):
	label = 'Tomato_healthy'
	y.append(label)
	path = os.path.join(Tomato_healthy,img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256,256))
	x.append([np.asarray(img)])
print(np.array(x).shape)

for img in tqdm(os.listdir(Tomato_Late_blight)[:200]):
	label = 'Tomato_Late_blight'
	y.append(label)
	path = os.path.join(Tomato_Late_blight,img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256,256))
	x.append([np.asarray(img)])
print(np.array(x).shape)

for img in tqdm(os.listdir(Tomato_Leaf_Mold)[:200]):
	label = 'Tomato_Leaf_Mold'
	y.append(label)
	path = os.path.join(Tomato_Leaf_Mold,img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256,256))
	x.append([np.asarray(img)])
print(np.array(x).shape)

for img in tqdm(os.listdir(Tomato_Septoria_leaf_spot)[:200]):
	label = 'Tomato_Septoria_leaf_spot'
	y.append(label)
	path = os.path.join(Tomato_Septoria_leaf_spot,img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256,256))
	x.append([np.asarray(img)])
print(np.array(x).shape)


for img in tqdm(os.listdir(Tomato_Spider_mites_Two_spotted_spider_mite)[:200]):
	label = 'Tomato_Spider_mites_Two_spotted_spider_mite'
	y.append(label)
	path = os.path.join(Tomato_Spider_mites_Two_spotted_spider_mite,img)
	img = cv2.imread(path)
	img = cv2.resize(img, (256,256))
	x.append([np.asarray(img)])
print(np.array(x).shape)

train_data = []
train_data = list(zip(x,y))
random.shuffle(train_data)

x1 = np.array([i[0] for i in train_data])
y1 = np.array([i[1] for i in train_data])

x = np.array(x)
y = np.array(y)

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(y1)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)

np_image_list = np.array(x1, dtype=np.float16) / 225.0

print("[INFO] Separando  los datos de entrenamiento, test")
x_train, x_test, y_train, y_test = train_test_split(x1, image_labels, test_size=0.2, random_state = 42)

aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")

model = Sequential()
inputShape = (largo_Imagen, ancho_Imagen, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, largo_Imagen, ancho_Imagen)
    chanDim = 1

model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))        

opt = Adam(lr=lr, decay=lr / EPOCAS)

model.compile(loss="binary_crossentropy", optimizer=opt,metrics=['accuracy'])
# train the network
print("[INFO] Entrenando red neuronal...")

history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=batch_size),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=EPOCAS, verbose=1
    )

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

print("[INFO] Calculando la precision del modelo")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")

print("[INFO] Saving model...")
pickle.dump(model,open('cnn_model.pkl', 'wb'))


model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")

#Loading the saved model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
lb = pickle.load(open(f"label_transform.pkl","rb"))

INIT_LR = 1e-3
opt=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
loaded_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = cv2.imread('img4.jpg')
img = cv2.resize(img,(256,256))
img = np.reshape(img,[1,256,256,3])

classes = loaded_model.predict_classes(img)

from scipy import sparse

sA = sparse.csr_matrix(classes)
lb = pickle.load(open(f"label_transform.pkl","rb"))
disease = f"{lb.inverse_transform(sA)[0]}"
print("The Disease detected is :",disease)

img2 = cv2.imread('img4.jpg')
plt.imshow(img2,)