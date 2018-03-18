import cv2
import numpy as np 
import os 

subjects = ["", "Yash Vig", "Ziyun He", "Devansh"]

def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    path = "/Users/Legend/anaconda2/lib/python2.7/site-packages/cv2/data/"
    face_cascade = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
 
    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
 
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None   
 
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
 
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        print(dir_name)
        if dir_name.isdigit():
               
            #------STEP-2--------
            #extract label number of subject from dir_name
            #format of dir name = slabel
            #, so removing letter 's' from dir_name will give us label
            label = int(dir_name)
            
            #build path of directory containing images for current subject subject
            #sample subject_dir_path = "training-data/s1"
            subject_dir_path = data_folder_path + "/" + dir_name
            
            #get the images names that are inside the given subject directory
            subject_images_names = os.listdir(subject_dir_path)
            
            #go through each image name, read image, 
            #detect face and add face to list of faces
            for image_name in subject_images_names:
            
                #ignore system files like .DS_Store
                if image_name.startswith("."):
                    continue
            
                #build image path
                #sample image path = training-data/s1/1.pgm
                image_path = subject_dir_path + "/" + image_name

                #read image
                image = cv2.imread(image_path)
            
                #detect face
                face, rect = detect_face(image)
            
                if face is not None:
                #add face to list of faces
                    faces.append(face)
                #add label for this face
                    labels.append(label)
            
            cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
        
    return faces, labels

faces, labels = prepare_training_data("dataset/train")
fr = cv2.face.LBPHFaceRecognizer_create()
fr.train(faces, np.array(labels))