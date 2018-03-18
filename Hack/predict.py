from learn import detect_face, fr, subjects
import cv2

def predict(test_img):
    #make a copy of the image as we don't want to change original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)
    
    # predict the image using our face recognizer 
    label = fr.predict(face)
    #get name of respective label returned by face recognizer
    return subjects[label[0]]


#load test images
test_img1 = cv2.imread("dataset/test/3/testdev.jpg")
 
#perform a prediction
ans = predict(test_img1)
data = {'Name' : ans}
import requests
import json 


print(ans)
