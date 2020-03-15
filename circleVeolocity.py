import numpy as np
import cv2
import time
import math
import imutils

"""
	Autor: Manuel Alejandro Medrano Díaz
	Materia: Visión artificial
	Tarea: Calcular la velocidad de una esfera
"""

#30 frames por segundo
vid = cv2.VideoCapture(0)

#colorMinimo = (20,100,100)
#colorMaximo = (30,255,255)
colorMinimo = (29, 86, 6)
colorMaximo = (64, 255, 255)

cont = 0
n = 30
inicial = (0,0)
tiempo = time.time()
velocidad = 0
radio = 1.8
pixs = 1

while(True):
	cont = cont+1
	ret, img = vid.read()
	blur = cv2.GaussianBlur(img, (11,11), 0)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

	#Mascara de color y ruido para obtencio de color en hsv
	mask = cv2.inRange(hsv, colorMinimo, colorMaximo)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	colors = cv2.bitwise_and(blur,blur, mask)
	blue_gray = cv2.cvtColor(colors, cv2.COLOR_BGR2GRAY)

	circles = cv2.HoughCircles(blue_gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=50, minRadius=20, maxRadius=70)
	#print(circles)
	if circles is not None:
		for i in circles[0,:]:
			# draw the outer circle
			cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
			# draw the center of the circle
			cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
			if(cont >= n):
				#Distancia euclidiana
				prod = math.pow(i[0]-inicial[0],2)+math.pow(i[1]-inicial[1],2)
				dis = math.sqrt(prod)
				#Calculo de distancia en cm, distancia euclidiana entre radio en pixeles por radio en cm
				dis = (dis/i[2])*radio
				#Tiempo
				resTime = (time.time() - tiempo)*30
				#velocidad
				velocidad = dis/resTime
				print("Velocidad:",velocidad)
				cont = 0
				#Punto de inicio del objeto
				inicial = (i[0],i[1])
				#Toma el tiempo actual
				tiempo = time.time()
	# Display the resulting frame
	cv2.imshow('detected circles',img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


#Terminar el uso de cámara.
vid.release()
cv2.destroyAllWindows()