
import cv2
import numpy as np
cap=cv2.VideoCapture(0)

face_c=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eyes_c=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_c.detectMultiScale(gray,1.1,10)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(w+x,h+y),(255,100,0),5)
        
        roi_gray=gray[y:h+y,x:x+w]
        roi=frame[y:h+y,x:x+w]
        eyes=eyes_c.detectMultiScale(roi_gray)
        for (a,b,c,d) in eyes:
            cv2.rectangle(roi,(a,b),(c+a,b+d),(0,255,0),4)
    cv2.imshow("abcd",frame)
    if(cv2.waitKey(1)==ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()