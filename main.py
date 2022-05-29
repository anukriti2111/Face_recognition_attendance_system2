import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

path = 'images'
images = []
RecordNames = []
testList = os.listdir(path)
for cu_img in testList:
    Test_image = cv2.imread(f'{path}/{cu_img}')
    images.append(Test_image)
    RecordNames.append(os.path.splitext(cu_img)[0])

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name):
    with open('Attendance Record.csv', 'r+') as f:
        DataList = f.readlines()
        nameList = []
        for line in DataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')

encodeListKnown = faceEncodings(images)
print('Encodings Complete')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    Test = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    Test = cv2.cvtColor(Test, cv2.COLOR_BGR2RGB)
    TestFrame = face_recognition.face_locations(Test)
    encodesTestFrame = face_recognition.face_encodings(Test, TestFrame)

    for encodeTest, TestLocation in zip(encodesTestFrame, TestFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeTest)
        TestDistance = face_recognition.face_distance(encodeListKnown, encodeTest)

        matchIndex = np.argmin(TestDistance)

        if matches[matchIndex]:
            name = RecordNames[matchIndex].upper()

            y1, x2, y2, x1 = TestLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()

