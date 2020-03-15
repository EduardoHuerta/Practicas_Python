import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers
import matplotlib.pyplot as plt

#this is the size of our encoded representations
encoding_dim = 32  #32 floats -> compression of factor 24.5, assuming the input is 784 floats

#this is our input placeholder
input_image = Input(shape=(784,))
# "encoded" is the encoded representation of the input 
encoded  = Dense(encoding_dim, activation='relu')(input_image)
# "decoded" is the lossy reconstruction of the input 
decoded =  Dense(784, activation='sigmoid')(encoded)
#this the model maps an input to its reconstruction
autoencoder = Model(input_image, decoded)  
#this the model maps an input to its encoded representation
encoder = Model(input_image, encoded)
#create a placeholder for an encoder (32-dimensiones) input
encoded_input = Input(shape=(encoding_dim,))
#retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
#create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
#configure our model to use a per-pixel binary- crosscentropy loss, and the Adadelta optimizher
autoencoder.compile(optimizer='adadelta', loss= 'binary_crossentropy')

#prepare out input data. Using MNIST digits
(x_train,_),(x_test,_) = mnist.load_data()
#normalize al values between 0 and 1 and we will flatten the 28x28 images ito vector of size 784
x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

#Entrenamos el autoencoder con 50 epocas
autoencoder.fit(x_train,x_train, epochs = 50, batch_size=256, shuffle= True, validation_data=(x_test,x_test))
#encode and decode some digits
#note that we take then from the test*set
encoded_images = encoder.predict(x_test)
decoded_images = decoder.predict(encoded_images)

#visualizamos la reconstruccion de la entrada y la representacion del encoded usando matplotlib
n= 20 #how many digits we will display
plt.figure(figsize=(20,4))
for i in range(n):
    #Mostramos la imagen original
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #muestra la reconstruida
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_images[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()    



