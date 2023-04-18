from innovative import app 
from flask import render_template,request
import cv2
import os
import numpy as np
import cvlib as cv
import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

global capture,rec_frame, grey, switch, neg, face, rec, out,camera
capture=0
switch=1
# camera=cv2.VideoCapture(cv2.CAP_DSHOW)


classes = ['man','woman']



@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html',val1=0) 



model = load_model('gender_detection.model')

def detect():
    frame=cv2.imread('innovative\static\img\shot.jpg')
    cv2.imwrite('innovative\static\img\shot.jpg',cv2.resize(frame, (350,400)))
    face, confidence = cv.detect_face(frame)
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
        print(conf)
        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        # cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
        #             2.5, (0, 0, 255), 5)

    # display output
    cv2.imwrite('innovative\static\img\\result.jpg',cv2.resize(frame, (350,400)))
    return idx, str(round(conf[idx] * 100, 2))

#################################################
@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if 'file1' in request.files:
            file1 = request.files['file1']
            pic_path=os.path.join(app.root_path,'static/img','shot.jpg')
            file1.save(pic_path)
        if request.form.get('pred') == 'Predict':
            id,conf=detect()
            global capture
            capture=0
            return render_template('home.html',val1=1,id=id,conf=conf)   
    elif request.method=='GET':
        return render_template('home.html',val1=0)
    return render_template('home.html', val1=0)