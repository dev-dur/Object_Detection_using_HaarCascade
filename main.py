import cv2 as cv 
import matplotlib.pyplot as plt 

config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'
thresh= 0.6
cap=cv.VideoCapture('test1.mp4')

if not cap.isOpened():
    cap=cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError('cannot open video')


# Video writer to save the output
output_filename = 'output_video.avi'
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))
#slowed = fps//4

out = cv.VideoWriter(output_filename, cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))


classlabels= []
file_name='coco.txt'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')


model = cv.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320,320)             #configurations are given in the documentation
model.setInputScale(1.0/127.5) 
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

while True:
    success,img=cap.read()
    #img = cv.resize(img, (954, 580))
    if not success:
        print("Failed to read frame")
        break  
    ClassIndex, confid, bbox = model.detect(img, confThreshold=thresh) 
    print(ClassIndex, bbox)
    if len(ClassIndex) != 0:
        for ClassInd, confidence, box in zip(ClassIndex.flatten(), confid.flatten(),bbox):
            if ClassInd <= 80:
                cv.rectangle(img,box,color=(0,255,0), thickness=1)
                cv.putText(img,classlabels[ClassInd-1].upper(),(box[0]+20,box[1]-5),cv.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)
                #cv.putText(img,str(round(confidence*100,2)),(box[0]+50,box[1]+20),cv.FONT_HERSHEY_COMPLEX,0.4,(0,255,0),1)
    out.write(img)
    cv.imshow('videodetection',img)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break 
cap.release()
out.release()
cv.destroyAllWindows()



