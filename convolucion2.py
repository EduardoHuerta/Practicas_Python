#Tarea convoluciones separables
#Eduardo Alfonso Huerta Mora
import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

'''
def convolucion(imagen,c):
    auxiliar = np.zeros((imagen.shape[0],imagen.shape[1]), dtype=np.int)
    for i in range(1,imagen.shape[0]-1):
        for j in range(1,imagen.shape[1]-1):
            valor = imagen.item(i-1, j-1)*c[0][0] + imagen.item(i-1,j)*c[0][1] + imagen.item(i-1,j+1)*c[0][2] + imagen.item(i,j-1)*c[1][0] + imagen.item(i+1,j-1)*c[2][0]
            if valor > 255:
                valor = 255
            elif valor < 0:
                valor = 0
            auxiliar.itemset((i,j),valor)
    return auxiliar
'''
#imagen = cv2.imread('C:/Users/eduar/Downloads/opencv-master/samples/data/lena.jpg',0)

#matriz = [[1,1,1],[0,0,0],[-1,-1,-1]]
#imagenR = convolucion(imagen,matriz)
#cv2.imshow('Bordes',imagenR)
#cv2.imwrite('bordesPrueba.bmp',imagenR)
# Tarea usando convoluciones separables.

img = cv.imread('C:/Users/eduar/Downloads/opencv-master/samples/data/lena.jpg')
img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

K = 11
p = np.int(np.floor(K/2))
Vlocal = np.zeros((K,K))

img_gray = np.pad(img_gray, p, mode='constant', constant_values=0)

size = np.shape(img_gray)

img_out1 = np.zeros((size[0],size[1]))
img_out2 = np.zeros((size[0],size[1]))

for x in range(p, size[0]): # filas
    #la p es para comenzaar el barrido con el kernel (p)
    if(x >= size[0] - p):
        continue
    for y in range(p, size[1]): # columnas
        if(y >= size[1] - p):
            continue
        Vleft = x-p # Restricciones para no salir de imagen para lado izquierdo
        Vright = x+p + 1 # Restricciones para no salir de imagen para lado derecho
        Vup = y-p # Restricciones para no salir de imagen para arriba
        Vdown = y+p + 1 # Restricciones para no salir de imagen para abajo
        Vlocal = img_gray[Vleft:Vright,Vup:Vdown] #tomamos la parte de la imagen correspondiente al kernel
        s, u, vh = np.linalg.svd(Vlocal) #matriz, vector, matriz
        img_out1[x,y] = np.median(np.dot(s * u, vh))
        img_out2[x,y] = np.median(Vlocal)

plt.subplot(121),plt.imshow(img2,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(img_out1,cmap = 'gray')
plt.title('Convoluciones Separables'), plt.xticks([]), plt.yticks([])

plt.show()
