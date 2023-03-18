import cv2
import time
from simple_facerec import SimpleFacerec

import socket

# Set the IP address and port number of the ESP8266
ip = '192.168.1.4'
port = 80

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the ESP8266
#s.connect((ip, port))

# Send a signal to turn on the LED






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

video=cv2.VideoCapture(0) # The IP address of the esp32 CAM
padding =20

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")


while True:
        
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
                        # Send signal to the ESP8266
                        if name == "shamel" or name == "amr" or name == "Mestekawy":
                            #s.send(b'1')
                            #time.sleep(1)
                            print("Known")
                        #else:
                            #s.send(b'0')
                    else:
                        # Perform age detection
                        ageNet.setInput(blob)
                        agePred=ageNet.forward()
                        age = agePred[0].argmax()

                        # Send signal to the ESP8266 if the age group is a baby
                        if age == 0:
                            #s.send(b'0')
                            print("baby")
                        else:
                            #s.send(b'1')
                            print("notbaby")

                    # Draw the bounding box and label on the frame
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
                    cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
                    end_time = time.time()
            fps = 1 / (end_time - start_time)
            cv2.putText(frame, f"FPS: {round(fps, 2)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    

 
            """
            for bbox,face_loc, name in zip(bboxs,face_locations, face_names):
                            #print (name)
                            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
                            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                            

                            ageNet.setInput(blob)
                            agePred=ageNet.forward()
                            age = agePred[0].argmax()
                                                    
                            if age == 0 and name == "Unknown":
                                s.send(b'1')                               
                            
                            if name != previous_name:
                                if name == "shamel":
                                    s.send(b'1')
                                    time.sleep(1)
                                    print("SHAMEL")
                            else:
                                s.send(b'0')

                            previous_name = name                                

                            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]                          
                            cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
                            #cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
                            cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
                            """

            cv2.imshow("Frame",frame)
            k=cv2.waitKey(1)
            if k==ord('q'):
                break

s.close()
video.release()
cv2.destroyAllWindows()

