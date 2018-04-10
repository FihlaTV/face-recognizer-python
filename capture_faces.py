import numpy as np
import cv2 as cv
from urllib.request import urlopen

CAM = 0
FRONTAL_FACE_CASCADE = 'haarcascades/haarcascade_frontalface_default.xml'
URL = 'http://192.168.137.173:8080/shot.jpg'

maxSamples = 20

def main ():
    faceDetect = cv.CascadeClassifier(FRONTAL_FACE_CASCADE)
    camera = cv.VideoCapture(CAM)
    sampleNum = 0

    id = input("Id: ")

    while True:
        ret, image = camera.read()
        #image = phoneCamera(URL) 
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        face = faceDetect.detectMultiScale(grayImage, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in face:
            sampleNum = sampleNum + 1
            cv.imwrite("faces/face." + str(id) + "." + str(sampleNum) + ".jpg", grayImage[y:y+h, x:x+w])
            cv.waitKey(delay=100)
            cv.rectangle(image, pt1=(x,y), pt2=(x+h,y+w), color=(255,255,0), thickness=2)

        cv.imshow("Capturando...", image)
        if sampleNum >= maxSamples: break
    
    camera.release()
    cv.destroyAllWindows()

def phoneCamera (url):
    imageResponse = urlopen(url)
    imageNp = np.array(bytearray(imageResponse.read()), dtype='uint8')
    image = cv.imdecode(imageNp, flags=-1)
    return image

if __name__ == '__main__':
    main()