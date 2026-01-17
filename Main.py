
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

main = tkinter.Tk()
main.title("Facial Emotion , Age and Gender Recognition") #designing main screen
main.geometry("1300x1100")

global filename
global X, Y
global face_classifier,age_classifier
genderProto = "model/deploy_gender.prototxt"
genderModel = "model/gender_net.caffemodel"
genderList = ['Male', 'Female']
ageList=['20','21','22','23','24','25','26','34','35','38','39','40','41','42','43']

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

face_emotion = ['angry','disgusted','fearful','happy','neutral','sad','surprised']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
gender_net = cv2.dnn.readNet(genderModel, genderProto)

def getGender(face):
    blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    genderPreds = gender_net.forward()
    gender = genderList[genderPreds[0].argmax()]
    return gender

def getID(name):
    index = 0
    for i in range(len(names)):
        if names[i] == name:
            index = i
            break
    return index        
    

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    
    
def processDataset():
    text.delete('1.0', END)
    global X, Y
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    text.insert(END,"Total number of images found in dataset is  : "+str(len(X))+"\n")
    text.insert(END,"Total facial expression found in dataset is : "+str(face_emotion)+"\n")
    

def trainFaceCNN():
    global face_classifier, age_classifier
    text.delete('1.0', END)
    if os.path.exists('model/cnnmodel.json'):
        with open('model/cnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            face_classifier = model_from_json(loaded_model_json)
        json_file.close()  

    if os.path.exists('model/cnnmodel_age.json'):
        with open('model/cnnmodel_age.json', "r") as json_file:
            loaded_model_json = json_file.read()
            age_classifier = model_from_json(loaded_model_json)
        json_file.close() 
  
        age_classifier.load_weights('model/cnnmodel_weights_age.h5')
        age_classifier._make_predict_function()

        face_classifier.load_weights("model/cnnmodel_weights.h5")
        face_classifier._make_predict_function()                  
    else:
        face_classifier = Sequential()
        face_classifier.add(Convolution2D(32, 3, 3, input_shape = (32, 32, 3), activation = 'relu'))
        face_classifier.add(MaxPooling2D(pool_size = (2, 2)))
        face_classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        face_classifier.add(MaxPooling2D(pool_size = (2, 2)))
        face_classifier.add(Flatten())
        face_classifier.add(Dense(output_dim = 256, activation = 'relu'))
        face_classifier.add(Dense(output_dim = 7, activation = 'softmax'))
        print(face_classifier.summary())
        face_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = face_classifier.fit(X, Y, batch_size=16, epochs=10, shuffle=True, verbose=2)
        face_classifier.save_weights('model/cnnmodel_weights.h5') 
        age_classifier.save_weights('model/cnnmodel_weights_age.h5')           
        model_json = face_classifier.to_json()
        with open("model/cnnmodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/cnnhistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    print(face_classifier.summary())
    f = open('model/cnnhistory.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[29] * 100
    text.insert(END,"CNN Training Model Accuracy = "+str(accuracy)+"\n\n") 

def predictFaceExpression():
    global face_classifier
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (32,32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = face_classifier.predict(img)
    predict = np.argmax(preds)

    preds1 = age_classifier.predict(img)
    predict1 = np.argmax(preds1)

    img = cv2.imread(filename)
    gender = getGender(img)
   
    img = cv2.resize(img, (600,400))
    cv2.putText(img, 'Facial Expression Recognized as : '+face_emotion[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.putText(img, 'Facial Gender Recognized as : '+gender, (10, 45),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.putText(img, 'Facial Age Recognized as : '+ageList[predict1], (10, 65),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    
    cv2.imshow('Facial Expression Recognized as : '+face_emotion[predict], img)
    cv2.waitKey(0)


def runWebCam():
    global face_classifier,age_classifier
    cap = cv2.VideoCapture(0)
    while True:
        _, img = cap.read()
        height, width, channels = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
        print("Found {0} faces!".format(len(faces)))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                sub_face = img[y:y+h, x:x+w]
            sub_face = cv2.resize(sub_face, (32,32))
            im2arr = np.array(sub_face)
            im2arr = im2arr.reshape(1,32,32,3)
            sub_face = np.asarray(im2arr)
            sub_face = sub_face.astype('float32')
            sub_face = sub_face/255
            preds = face_classifier.predict(sub_face)
            predict = np.argmax(preds)
            gender = getGender(img)
            preds1 = age_classifier.predict(sub_face)
            predict1 = np.argmax(preds1)
            
            cv2.putText(img, 'Facial Expression Recognized as : '+face_emotion[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
            cv2.putText(img,"Gender detected as "+gender, (10, 45),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
            cv2.putText(img,"Age detected as "+ageList[predict1], (10, 65),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
            cv2.imshow('output', img)
            
            
        if cv2.waitKey(650) & 0xFF == ord('q'):
            break   
    cap.release()
    cv2.destroyAllWindows()


def graph():
    f = open('model/cnnhistory.pckl', 'rb')
    cnn_data = pickle.load(f)
    f.close()
    face_accuracy = cnn_data['accuracy']
    face_loss = cnn_data['loss']

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Iterations/Epoch')
    plt.ylabel('Accuracy')
    plt.plot(face_accuracy, 'ro-', color = 'green')
    plt.plot(face_loss, 'ro-', color = 'orange')
    plt.legend(['Face Accuracy', 'Face Loss'], loc='upper left')
    plt.title('CNN Accuracy Comparison Graph')
    plt.show()

def exit():
    main.destroy()

font = ('times', 13, 'bold')
title = Label(main, text='Facial Emotion , Age and Gender Recognition ')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=480,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Facial Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

cnnButton = Button(main, text="Train CNN Algorithm", command=trainFaceCNN)
cnnButton.place(x=50,y=200)
cnnButton.config(font=font1) 

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=50,y=250)
graphButton.config(font=font1)

predictfaceButton = Button(main, text="Predict Facial Emotion", command=predictFaceExpression)
predictfaceButton.place(x=50,y=300)
predictfaceButton.config(font=font1)

webcamButton = Button(main, text="Predict Facial Emotion from WEBCAM", command=runWebCam)
webcamButton.place(x=50,y=350)
webcamButton.config(font=font1) 

main.config(bg='#87CEEB')
main.mainloop()
