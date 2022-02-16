import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import cv2 as cv
import pytesseract

#####Read the model Weights
def initializePredectionModel():
    model=load_model('cnn-mnist-new.h5')
    return model

# split the board into 81 individual images
def split_boxes(img):
    rows=np.vsplit(img,9)
    boxes=[]
    for i,r in enumerate(rows):
        cols=np.hsplit(r,9)
        for j,box in enumerate(cols):
            boxes.append(box)
            # winname = str(i*9+j)
            # cv.imshow(winname,box)
            # winname=winname+'.jpg'
            # cv.imwrite()
    return boxes

def reorder(myPoints):
    myPoints=myPoints.reshape((4,2))
    myPointsNew=np.zeros((4,1,2),dtype=np.int32)
    add=myPoints.sum(1)
    myPointsNew[0]=myPoints[np.argmin(add)]
    myPointsNew[3]=myPoints[np.argmax(add)]
    diff=np.diff(myPoints,axis=1)
    myPointsNew[1]=myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def biggestContour(contours):
    biggest=np.array([])
    max_area=0
    for i in contours:
        area=cv.contourArea(i)
        if area>50:
            peri=cv.arcLength(i,True)
            approx=cv.approxPolyDP(i,0.02*peri,True)
            if area> max_area and len(approx)==4:
                biggest=approx
                max_area=area
    return biggest,max_area

def getPredection(boxes,model):
    result=[]
    for image in boxes:
        img=np.asarray(image)
        img = img[:,:,0]
        img=img[4:img.shape[0]-4,4:img.shape[1]-4]
        img=cv.resize(img,(28,28))
        img=img/255
        img=img.reshape(1,28,28,1)

        #Get Prediction

        predictions=model.predict(img)
        #classIndex=model.predict_classes(img)
        classIndex=np.argmax(predictions,axis=-1)
        probabilityValue=np.amax(predictions)
        # print(classIndex,probabilityValue)

        ##Save to Result
        if probabilityValue>0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

def displayNumbers(img,numbers,color=(0,255,0)):
    secW=int(img.shape[1]/9)
    secH=int(img.shape[0]/9)
    for x in range(0,9):
        for y in range(0,9):
            if numbers[(y*9)+x]!=0:
                cv.putText(img,str(numbers[(y*9)+x]),(x*secW+int(secW/2)-10,int((y+0.8)*secH)), cv.FONT_HERSHEY_SIMPLEX,2,color,2,cv.LINE_AA)
    return img

def drawGrid(img):
    secW=int(img.shape[1]/9)
    secH=int(img.shape[0]/9)
    for i in range(0,9):
        pt1=(0,secH*i)
        pt2=(img.shape[1],secH*i)
        pt3=(secW*i,0)
        pt4=(secW*i,img.shape[0])
        cv.line(img,pt1,pt2,(255,255,0),2)
        cv.line(img,pt3,pt4,(255,255,0),2)
    return img








