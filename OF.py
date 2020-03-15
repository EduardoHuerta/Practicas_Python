import numpy as np
import cv2

# Iniciamos la webcam
cap = cv2.VideoCapture(0)

# Toma el primer frame y lo convierte en escala de grises
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Get the Shi Tomasi corners to use them as initial reference points
corners = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=500, qualityLevel=0.3, minDistance=0, blockSize=7)
cornerColors = np.random.randint(0, 255, (corners.shape[0], 3))

# Crear una imagen de máscara para dibujar
mask = np.zeros_like(frame)

# Definimos los parametros para el flujo optico Lucas Kanade
lkParameters = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03))

# Reproduce hasta que el usuario decide parar
while True:
    # Guardamos los datos del frame previo
    previousGray = gray
    previousCorners = corners.reshape(-1, 1, 2)

    # Obtenemos el siguiente frame
    ret , frame = cap.read()

    if ret:
        # Convertimos el frame a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculamos el flujo optico
        corners, st, err = cv2.calcOpticalFlowPyrLK(previousGray, gray, previousCorners, None, **lkParameters)
        
        # Seleccionamos unicamente los buenos corners
        corners = corners[st == 1]
        previousCorners = previousCorners[st == 1]
        #cornerColors[st == 1]

        # Compruebe que todavía quedan algunos corners
        if corners.shape[0] == 0:
            print('Stopping. There are no corners left to track')
            break
        
        # Dibuja los corner tracks
        for i in range(corners.shape[0]):
            x, y = corners[i]
            xPrev, yPrev = previousCorners[i]
            color = cornerColors[i].tolist()
            frame = cv2.circle(frame, (x, y), 5, color, -1)
            mask = cv2.line(mask, (x, y), (xPrev, yPrev), color, 2)
        frame = cv2.add(frame, mask)
        
        # Mostramos el resultado del frame
        cv2.imshow('Flujo optico', frame)
        k = cv2.waitKey(30) & 0xff

         # Salir si el usuario presiona ESC
        if k == 27:
            break
    else:
        break

# Cuando todo esta hecho, eliminamos la captura y cerramos las ventanas mostradas
cap.release()
cv2.destroyAllWindows()    
