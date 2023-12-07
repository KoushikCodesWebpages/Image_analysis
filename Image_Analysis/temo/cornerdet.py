import cv2
import numpy as np
img=cv2.imread("bts wp 1.jpeg")
img=cv2.resize(img,(750,500))
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners=cv2.goodFeaturesToTrack(gray,150,0.2,10)
corners=np.int16(corners)
for corner in corners:
    x,y=corner.ravel()
    cv2.circle(img,(x,y),5,(0,255,0),-1)

cv2.imshow("Hi",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
