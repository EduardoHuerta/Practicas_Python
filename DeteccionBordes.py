import cv2 as cv

cap = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while(True):
    ret, frame = cap.read()
    #gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv.imshow('Frame', frame)
    #cv2.imshow('Gris', gray)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
