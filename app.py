import cv2
import time
from simple_facerec import SimpleFacerec

import socket

# Set the IP address and port number of the ESP8266
"""
ip = '192.168.1.4'
port = 80

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the ESP8266
s.connect((ip, port))


from twilio.rest import Client

account_sid = 'ACe22c52b972c41a137d3e0d55481a7990'
auth_token = '4478c65476229e32ff59a16cc953cefe'
client = Client(account_sid, auth_token)

whatsapp_number = 'whatsapp:+201006623926' 




"""





def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            #cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame, bboxs


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"



faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
URL = "http://192.168.1.103:81/stream"

video=cv2.VideoCapture(0) 
padding =20

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

last_check_time = time.time() - 5  # initialize last check time to 5 seconds ago


count_not_unknown = 0
count_unknown=0
while True:
        current_time = time.time()
        time_since_last_check = current_time - last_check_time
        success, frame =video.read()

        if not success:
            break
        else:
            start_time = time.time()

            # Detect Faces

            face_locations, face_names = sfr.detect_known_faces(frame)
            frame,bboxs=faceBox(faceNet,frame)
            for bbox,face_loc, name in zip(bboxs,face_locations, face_names):
                    face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
                    blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

                    # Check if the face is known or unknown
                    if name != "Unknown":
                        count_not_unknown +=1
                    else:
                        count_not_unknown= 0
                    if count_not_unknown ==20:
                        print("known")
                        count_not_unknown =0
                        """
                        s.send(b'1')
                       
                        if time_since_last_check >= 5:
                            # Send message and update the last_detection dictionary for this person
                            message = client.messages.create(
                                body=f'{name} face is detected!',
                                from_='whatsapp:+14155238886', 
                                to=whatsapp_number
                            )
                            last_check_time = current_time
                            """

                    if name == "Unknown":
                            count_unknown+=1
                            # Perform age detection
                            ageNet.setInput(blob)
                            agePred=ageNet.forward()
                            age = agePred[0].argmax()
                    else:
                         count_unknown=0

                    if count_unknown == 20:
                            count_unknown= 0
                            if age == 0:
                                #s.send(b'0')
                                print("baby")
                            else:
                                #s.send(b'1')
                                print("not baby")
                                """
                                if time_since_last_check >= 5:
                                    message = client.messages.create(
                                body=f'{name} face is detected!',
                                from_='whatsapp:+14155238886', 
                                to=whatsapp_number
                                )
                                last_check_time = current_time
                                """


                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                    cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f"FPS: {round(fps, 2)}", (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,  255,0), 2)


            cv2.imshow("Frame",frame)
            k=cv2.waitKey(1)
            if k==ord('q'):
                break

#s.close()
video.release()
cv2.destroyAllWindows()

