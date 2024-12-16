import cv2
import numpy as np
import os

from cv2.data import haarcascades





stopsign=cv2.CascadeClassifier("Haar_stopsign.xml")

resimler=[r"C:\Users\Yusuf\PycharmProjects\PythonProjectcv\DURMAFOTO\foto1.jpg",
    r"C:\Users\Yusuf\PycharmProjects\PythonProjectcv\DURMAFOTO\foto2.jpg",
    r"C:\Users\Yusuf\PycharmProjects\PythonProjectcv\DURMAFOTO\foto3.jpg",
    r"C:\Users\Yusuf\PycharmProjects\PythonProjectcv\DURMAFOTO\foto4.jpg",
    r"C:\Users\Yusuf\PycharmProjects\PythonProjectcv\DURMAFOTO\foto5.jpg",]

haar_cascade=cv2.CascadeClassifier("Haar_stopsign.xml")



for foto in resimler:
    img=cv2.imread(foto)
    fotogray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sign_detect=haar_cascade.detectMultiScale(fotogray,scaleFactor=1.1,minNeighbors=8)
    for (x,y,w,h) in sign_detect:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),thickness=2)



    cv2.imshow("karelenmiş", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
