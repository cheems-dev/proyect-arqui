import cv2
import numpy as np
import HandTrackingModule as htm
import time
frameR = 100
smoothening = 5

import autopy

wCam, hCam = 640, 480
pTime = 0
plocX,polcY = 0,0
clocX,clocY = 0,0

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
print(wScr,hScr)

while True:
    #Para buscar las marcas en las manos
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    #print(lmList)

    #Para ver que dedos estan cerrados
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #print(x1,x2,y1,y2)

        fingers = detector.fingersUp()

        #Solo si esta el primer dedo se movera el cursor
        if fingers[1]==1 and fingers[2]==0:
            #Convertir coordenadas
            cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
            x3 = np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y1, (frameR,hCam-frameR),(0,hScr))

            #Suavizar el movimiento del cursor
            clocX = plocX + (x3 - plocX)/smoothening
            clocY = polcY + (y3-polcY)/smoothening
            #Mover el mouse
            autopy.mouse.move(clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocX,polcY = clocX,clocY

        #Para hacer click con los dedos
        if fingers[1] == 1 and fingers[2] == 1:
            length,img,lineInfo = detector.findDistance(8,12,img)
            #print(length)
            if length<35:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()
        #print(fingers)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)


