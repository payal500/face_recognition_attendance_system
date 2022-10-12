import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
path = 'ImageAttendance'
images = []
ClassNames = []
myList = os.listdir(path)


for cls in myList:
    curImg=cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    ClassNames.append(os.path.splitext(cls)[0])


def findencodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode= face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
def markAttendance(name):
    with open('attendence.csv','r+') as f:
        myDataList=f.readlines()
        namelist=[]
        for line in myDataList:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            dtString2=now.date()
            f.writelines(f'\n{name},{dtString},{dtString2}')

encodeListKnown=findencodings(images)
print('encoding comlete')


cap=cv2.VideoCapture(0)

while True:
    sucess, img=cap.read()

    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame= face_recognition.face_locations(imgS)
    encodeCurFrame= face_recognition.face_encodings(imgS,facesCurFrame)
    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)

        matchindex=np.argmin(faceDis)
        if matches[matchindex]:
            name = ClassNames[matchindex].upper()

            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
            cv2.imshow('webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

