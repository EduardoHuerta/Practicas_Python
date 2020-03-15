import cv2 as cv
import numpy as np

camara = cv.VideoCapture(0)
#color azul
low_blue = np.array([94,81,4])
high_blue = np.array([126,255,255])

while(True):
    ret, frame = camara.read()
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #mascara para ver solo azules
    mascara_azul = cv.inRange(hsv_frame,low_blue, high_blue)    
    #azul = cv.bitwise_and(frame, frame,mask=mascara_azul)

    #kernel = np.ones((5,5),np.uint8)
    #erosion = cv.erode(mascara_azul,kernel,iterations = 1)
    #dilatacion = cv.dilate(mascara_azul,kernel,iterations = 1)

    #opening = cv.morphologyEx(mascara_azul, cv.MORPH_OPEN, kernel)
    #closing = cv.morphologyEx(mascara_azul, cv.MORPH_CLOSE, kernel)

    # cv.imshow('Frame', dilatacion)
    #cv.imshow('Opening',opening)
    cv.imshow('Frame original',frame)
    #Imagen segmentada
    frame[mascara_azul != 0] = [10, 255, 10]
    cv.imshow('Imagen que cambia las tonalidades azules por verdes', frame)
    #cv.imshow('Mascara', azul)


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv.destroyAllWindows()
