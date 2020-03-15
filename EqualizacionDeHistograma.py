import cv2
import numpy as np
from matplotlib import pyplot as plt 

img1 = cv2.imread('cr7.jpg',0) 
cv2.imshow('Imagen original',img1)
#Genera el historama de la imagen 
hist,bins = np.histogram(img1.flatten(),256,[0,256])
#Genera la función de distribución acumulada (cdf por sus siglas en inglés) 
cdf = hist.cumsum() 
cdf_normalized = cdf * hist.max()/ cdf.max() 
#Genera los gráficos del histograma y de la función de distribución acumulada
plt.plot(cdf_normalized, color = 'b') 
plt.hist(img1.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histograma'), loc = 'upper right') 
plt.show()
#Enmascara los valores iguales a cero
cdf_m = np.ma.masked_equal(cdf,0)    
# #Aplica la transformación de ecualización
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())   
#Rellena los valores previamente enmascarados con ceros - 
cdf = np.ma.filled(cdf_m,0).astype('uint8')   
#Aplica la ecualización a los píxeles de la imagen original - 
img2 = cdf[img1]   
#Grafica la imagen resultante de aplicar la ecualización del histograma - 
cv2.imshow('imagen ecualizada',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()     
#img = cv2.imread('cr7.jpg',0) 
#equ = cv2.equalizeHist(img) 
#res = np.hstack((img,equ)) #Agrupa las imágenes una al lado de la otra - 
#cv2.imshow('Imagen ecualizada',res)
#cv2.waitKey(0) 
#cv2.destroyAllWindows()