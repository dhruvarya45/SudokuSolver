import cv2
import keras.models

print('Setting UP')
import os
import pytesseract
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# from tensorflow.keras.models import load_model
import numpy as np
import imutils
# from imutils.perspective import four_point_transform
from utlis import *
import SudokuSolver
import cv2 as cv
pytesseract.pytesseract.tesseract_cmd='Tesseract-OCR\\tesseract.exe'

# model=initializePredectionModel()
model = keras.models.load_model('ocr.h5')
# 1. Preparing the image
img=cv.imread('3.jpeg')
img=cv.resize(img,(450,450),interpolation=cv.INTER_CUBIC)
cv.imshow("Image", img)

# print(pytesseract.image_to_boxes(img))

imgBlank=np.zeros((450,450,3),np.uint8)
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgBlur=cv.GaussianBlur(gray,(5,5),1)
# cv.imshow("Blurred Image", imgBlur)
imgThreshold=cv.adaptiveThreshold(imgBlur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,4)
cv.imshow("Threshold Image", imgThreshold)
print(img.shape)
print(imgThreshold.shape)

# 2. Finding all contours

imgContours=imgThreshold.copy()
imgBigContour=imgThreshold.copy()
contours, hierarchies = cv.findContours(imgThreshold, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(imgContours, contours, -1, (0,255,255),1)

# 3.Find the biggest contour
biggest,maxArea=biggestContour(contours)
print(biggest)
if biggest.size!=0:
    biggest=reorder(biggest)
    cv.drawContours(imgBigContour,biggest,-1,(0,0),25)
    pts1=np.float32(biggest)
    pts2=np.float32([[0,0],[450,0],[0,450],[450,450]])
    matrix=cv.getPerspectiveTransform(pts1,pts2)
    imgWrapColored=cv.warpPerspective(img,matrix,(450,450))
    imgDetectedDigits=img.copy()
    # imgWrapColored=cv.cvtColor(imgThreshold,cv.COLOR_BGR2GRAY)
    cv.imshow("Biggest Contour", imgBigContour)
    cv.imshow("Image", imgWrapColored)

   #4. Split the Image and find each digit available
    imgSolvedDigits=imgWrapColored.copy()
    boxes=split_boxes(imgWrapColored)
    print(len(boxes))
    cv.imshow("Sample", boxes[71])
    numbers=getPredection(boxes,model)
    print(numbers)
    imgDetectedDigits=displayNumbers(imgDetectedDigits,numbers,color=(255,0,255))
    # cv.imshow("Image Detected Digits", imgDetectedDigits)
    numbers=np.asarray(numbers)
    posArray=np.where(numbers > 0,0,1)
    print(posArray)

    ####5.Find the solution of the sudoku board
    board=np.array_split(numbers,9)
    try:
        sudokuSolver.solve(board)
    except:
        pass
    print(board)
    flatList=[]
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    solvedNumbers=flatList*posArray
    imgSolvedDigits=displayNumbers(imgSolvedDigits,solvedNumbers)

    ####6. Overlay Solution
    pts2=np.float32(biggest)
    pts1=np.float32([[0,0],[450,0],[0,450],[450,450]])
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    imgInvWarpColored=img.copy()
    imgInvWarpColored=cv.warpPerspective(imgSolvedDigits,matrix,(450,450))
    inv_perspective=cv.addWeighted(imgInvWarpColored,1,img,0.5,1)
    imgDetectedDigits=drawGrid(imgDetectedDigits)
    imgSolvedDigits=drawGrid(imgSolvedDigits)

    # imageArray=([img,imgThreshold,imgContours,imgBigContour],[imgDetectedDigits,imgSolvedDigits,imgInvWarpColored,inv_perspective])
    # stackedImage=stackImages(imageArray,1)
    # cv.imshow('Stacked Images', inv_perspective)

else:
   print("No Sudoku Found")

cv.waitKey(0)

