import cv2
import numpy as np
import face_recognition
imgobama=face_recognition.load_image_file('ImageAttendance/obama.jpg')
imgobama=cv2.cvtColor(imgobama,cv2.COLOR_BGR2RGB)
imgobama2=face_recognition.load_image_file('ImageAttendance/obama2.jpg')
imgobama2=cv2.cvtColor(imgobama2,cv2.COLOR_BGR2RGB)

faceLoc= face_recognition.face_locations(imgobama)[0]
encodeobama=face_recognition.face_encodings(imgobama)[0]
cv2.rectangle(imgobama,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocobama2= face_recognition.face_locations(imgobama2)[0]
encodeobama2=face_recognition.face_encodings(imgobama2)[0]
cv2.rectangle(imgobama2,(faceLocobama2[3],faceLocobama2[0]),(faceLocobama2[1],faceLocobama2[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodeobama],encodeobama2)
faceDis=face_recognition.face_distance([encodeobama],encodeobama2)
print(results,faceDis)
cv2.putText(imgobama2,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('obama',imgobama)
cv2.imshow('obama2',imgobama2)
cv2.waitKey(0)



