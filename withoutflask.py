import cv2
from simple_facerec import SimpleFacerec
import time

 

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
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame, bboxs


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"



faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)

faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


video=cv2.VideoCapture(0)


baby_detected = False
baby_detected_start_time = 0

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

padding=50
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 # generate frames for web display
while True:
        success, frame =video.read()

        if not success:
            break
        else:
            # Detect Faces
            face_locations, face_names = sfr.detect_known_faces(frame)
            frame,bboxs=faceBox(faceNet,frame)
 

            for bbox in bboxs:
                            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
                            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                            


                            ageNet.setInput(blob)
                            agePred=ageNet.forward()
                            age = agePred[0].argmax()
                        
                            if age == 0:  # if age is between 0-2
                                label = "baby"
                                if not baby_detected:
                                    baby_detected = True
                                    baby_detected_start_time = time.time()

                                elif time.time() - baby_detected_start_time >= 1:
                                    message = "Baby detected for more than 1 second!"
                            else:
                                label = 'Not Baby'
                                baby_detected = False
                                baby_detected_start_time = 0

                            label = "{}".format(label)

                            cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
                            cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)

            for face_loc, name in zip(face_locations, face_names):
                        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

            cv2.imshow("Age",frame)
            k=cv2.waitKey(1)
            if k==ord('q'):
                break
video.release()
cv2.destroyAllWindows()



