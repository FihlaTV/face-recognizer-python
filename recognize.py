import numpy as np
import cv2 as cv
from urllib.request import urlopen

CAM = 0
URL = 'http://192.168.137.173:8080/shot.jpg'
FRONTAL_FACE_CASCADE = 'haarcascades/haarcascade_frontalface_default.xml'
DATA = 'train_data.yml'


def main ():
    faceDetect = cv.CascadeClassifier(FRONTAL_FACE_CASCADE)
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read(DATA)
    font = cv.FONT_HERSHEY_COMPLEX_SMALL
    camera = cv.VideoCapture(CAM)

    while True:
        ret, image = camera.read()
        #image = phoneCamera(URL)
        grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        face = faceDetect.detectMultiScale(grayImage, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in face:
            cv.rectangle(image, pt1=(x,y), pt2=(x+w,y+w), color=(255,255,0), thickness=2)
            id, conf = recognizer.predict(grayImage[y:y+h, x:x+w])

            if int(id) == 1: faceName = "Ivan"
            if int(id) == 2: faceName = "Johnny"
            cv.putText(image, faceName, org=(x,y), fontFace=font, fontScale=1.5, color=(255,255,0), thickness=2)

        cv.imshow("Rostros", image)
        if cv.waitKey(delay=1) == ord('q'):break
    
    camera.release()
    cv.destroyAllWindows()

def phoneCamera (url):
    imageResponse = urlopen(url)
    imageNp = np.array(bytearray(imageResponse.read()), dtype='uint8')
    image = cv.imdecode(imageNp, flags=-1)
    return image

if __name__ == '__main__':
    main()