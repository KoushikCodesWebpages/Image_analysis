import cv2
import numpy as np
base=cv2.imread("bts wp 1.jpeg")
tem=cv2.imread("yoongi.png")
gray_b=cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)
gray_t=cv2.cvtColor(tem,cv2.COLOR_BGR2GRAY)
h1,w1=gray_t.shape
methods=[cv2.TM_CCOEFF,
         cv2.TM_CCOEFF_NORMED,
         cv2.TM_CCORR,
         cv2.TM_CCORR_NORMED,
         cv2.TM_SQDIFF,
         cv2.TM_SQDIFF_NORMED]
for method in methods:
    result=cv2.matchTemplate(gray_b,gray_t,method)
    min,max,minloc,maxloc=cv2.minMaxLoc(result)
    
    if(method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]):
        location=minloc
    else:
        location=maxloc
    dist=(w1+location[0],h1+location[1])
    cv2.rectangle(base,location,dist,(0,0,0),5)

    base=cv2.resize(base,(750,500))
    cv2.imshow("Hi",base)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
