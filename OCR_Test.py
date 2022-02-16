import numpy as np
import cv2 as cv
import pickle

##############################
width=640
height=480
##############################

cap=cv.VideoCapture(1);
cap.set(3,width)
cap.set(4,height)

pickle_in=open("model_trained.p","rb")
model=pickle.load(pickle_in)

def preProcessing(img):
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img=cv.equalizeHist(img)
    img=img/255
    return img

while True:
    success, imgOriginal=cap.rea()
    img=np.asarray(imgOriginal)
    img=cv.resize(img(32,32))
    img=preProcessing(img)
    cv.imshow("Preprocessed Image", img)
    img=img.reshape(1,32,32,1)
    #Predict
    classIndex=int(model.predict_classes(img))
    print(classIndex)

    if cv.waitKey(1) & 0xFF==ord('q'):
        break