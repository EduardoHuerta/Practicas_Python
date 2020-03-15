import cv2 as cv
import numpy as np

camara = cv.VideoCapture(0)
kernel = np.ones((5,5), np.uint8)
while(True):
    ret, frame = camara.read()
    rangoMax = np.array([50, 255, 50])
    rangoMin = np.array([0, 51, 0])
    mascara = cv.inRange(frame, rangoMin, rangoMax)
    opening = cv.morphologyEx(mascara, cv.MORPH_OPEN, kernel)
    x,y,w,h = cv.boundingRect(opening)
    cv.rectangle(frame,(x,y),(x+w, y+h),(0,255,0),3)
    cv.circle(frame,(int(x+w/2),int(y+h/2)),5,(0,0,255),-1)
    cv.imshow('Camara',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
camara.release()
cv.destroyAllWindows()
