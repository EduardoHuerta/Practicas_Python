#PRE PROCESAMIENTO DE IMAGENES
#Escalado de caracteristicas
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Importamos paquetes de keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

#Inicializando Red Neuronal Convolucional
clasificador = Sequential()

#Paso 1 - Convolucion
clasificador.add(Conv2D(input_shape=(64,64,3), filters=32, kernel_size=3, strides=3,activation='relu'))

#Paso 2 - Agrupacion mediante MaxPooling
clasificador.add(MaxPooling2D(pool_size=(3,3)))

#Paso 3 - Aplanamiento mediante Flatten
clasificador.add(Flatten())

#Paso 4 - Conexion Completa
clasificador.add(Dense(units=128, activation='relu'))
clasificador.add(Dense(units=1, activation='sigmoid'))

#Paso 5 - Compilacion
clasificador.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Encajar la red neuronal en las imagenes
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)