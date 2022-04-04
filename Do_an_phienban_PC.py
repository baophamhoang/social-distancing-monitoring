import numpy as np
import argparse
import cv2
import time
from math import pow, sqrt
from playsound import playsound

# Xu ly tham so dau vao
parser = argparse.ArgumentParser(description='Use MobileNet SSD on Pi for object detection')
parser.add_argument("--vid_file", help="Duong dan den file video")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt")
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel")
args = parser.parse_args()

# Cac nhan cua network
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

def position(detections,i,cols,rows):
    # Lay class_id
    class_id = int(detections[0, 0, i, 1])

    # Tinh toan vi tri cua doi tuong
    xLeftBottom = int(detections[0, 0, i, 3] * cols)
    yLeftBottom = int(detections[0, 0, i, 4] * rows)
    xRightTop = int(detections[0, 0, i, 5] * cols)
    yRightTop = int(detections[0, 0, i, 6] * rows)


    heightFactor = frame.shape[0] / 300.0
    widthFactor = frame.shape[1] / 300.0


    xLeftBottom = int(widthFactor * xLeftBottom)
    yLeftBottom = int(heightFactor * yLeftBottom)
    xRightTop = int(widthFactor * xRightTop)
    yRightTop = int(heightFactor * yRightTop)

    return class_id, xLeftBottom, yLeftBottom, xRightTop, yRightTop
F=625
def do_detect(frame, net, classNames):

    # Resize anh ve 300x300
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    detections = net.forward()
    # Xu ly output cua mang
    cols = frame_resized.shape[1]
    rows = frame_resized.shape[0]
    # Duyet qua cac object detect duoc
    for i in range(detections.shape[2]):
        # Lay gia tri confidence
        confidence = detections[0, 0, i, 2]
        # Neu vuot qua 0.5 threshold
        if confidence > 0.5:
            # Tinh toan vi tri cua doi tuong
            class_id, xLeftBottom, yLeftBottom, xRightTop, yRightTop = position(detections, i, cols, rows)
            # Ve label cua doi tuong
            if class_id == 15:
                # Ve khung hinh chu nhat
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0))
                label = "Confidence: " + str(confidence)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                x_mid = round((xLeftBottom + xRightTop) / 2, 4)
                y_mid = round((yLeftBottom + yRightTop) / 2, 4)
                height = round(yRightTop - yLeftBottom, 4)
                # Doi sang cm
                distance = (165 * F) / height
                x_mid_cm = (x_mid * distance) / F
                y_mid_cm = (y_mid * distance) / F
                DStoado.append([x_mid_cm, y_mid_cm, distance,i,xLeftBottom, yLeftBottom])
    return  frame
def gan_nhau(x, y):
    Khoangcach= sqrt(pow(x[0]-y[0],2)+pow(x[1]-y[1],2)+pow(x[2]-y[2],2))
    #print(Khoangcach)
    kt=2
    if (Khoangcach<300) and (Khoangcach>200):
        kt=1
    if (Khoangcach<200) and (Khoangcach>0):
        kt=0
    return Khoangcach

# Mo video hoac webcam
if args.vid_file:
    cap = cv2.VideoCapture(args.vid_file)
else:
    cap = cv2.VideoCapture(0)

# Load model
net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)

# Bat dau doc tu video/webcam
# Bien dem frame
i_frame = 0
frame2=0
warning = 0
dem_warning = 0
while True:

    DStoado = []
    start=time.time()
    # Doc tung frame
    ret, frame = cap.read()
    # Tang bien dem
    i_frame +=1
    # Xu ly detection moi 20 frame de giam tai cho Pi
        # Thuc hien detect
    frame = do_detect(frame,net,classNames)
    if len(DStoado)>1:
        for i in range(len(DStoado)):
            for j in range(i+1,len(DStoado)):
                if gan_nhau(DStoado[i],DStoado[j])<300:

                    x_i_centroid= round(DStoado[i][0]*F/DStoado[i][2])
                    x_j_centroid= round(DStoado[j][0]*F/DStoado[j][2])
                    y_i_centroid= round(DStoado[i][1]*F/DStoado[i][2])
                    y_j_centroid= round(DStoado[j][1]*F/DStoado[j][2])

                    if (gan_nhau(DStoado[i],DStoado[j])<200) and (gan_nhau(DStoado[i],DStoado[j])>0):
                        mau=([0,0,255])
                        # toa do frame i
                        i_xLeftBottom = DStoado[i][4]
                        i_yLeftBottom = DStoado[i][5]
                        i_xRightTop = x_i_centroid * 2 - i_xLeftBottom
                        i_yRightTop = y_i_centroid * 2 - i_yLeftBottom
                        cv2.rectangle(frame, (i_xLeftBottom, i_yLeftBottom), (i_xRightTop, i_yRightTop), (0, 0, 255), 2)
                        # toa do frame j
                        j_xLeftBottom = DStoado[j][4]
                        j_yLeftBottom = DStoado[j][5]
                        j_xRightTop = x_j_centroid * 2 - j_xLeftBottom
                        j_yRightTop = y_j_centroid * 2 - j_yLeftBottom
                        cv2.rectangle(frame, (j_xLeftBottom, j_yLeftBottom), (j_xRightTop, j_yRightTop), (0, 0, 255), 2)
                        if i_frame-frame2>7:
                            playsound('pippip.wav',False)
                            frame2=i_frame
                            warning = 1
                    if gan_nhau(DStoado[i],DStoado[j])>200:
                        mau=([0,255,255])
                    cv2.line(frame, (x_i_centroid, y_i_centroid), (x_j_centroid, y_j_centroid), mau , 1)
    if warning == 1:
        frame_size = cv2.getWindowImageRect('frame')
        (label_width, label_height), baseline = cv2.getTextSize("Chu y khoang cach 2m", cv2.FONT_HERSHEY_DUPLEX, 1, 1)
        dem_warning = dem_warning + 1

        frame_x, frame_y =  (frame_size[2]//2 - label_width//2 , frame_size[3]//6 )
        cv2.putText(frame, "Chu y khoang cach 2m", (frame_x, frame_y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
        cv2.rectangle(frame, (frame_x - 10 , frame_y+ 10 ), (frame_x + label_width + 10, frame_y - label_height - 10), (0, 0, 255), 2)
        if dem_warning > 20:
            warning = 0
            dem_warning = 0
        # Hien thi frame len man hinh
    end=time.time()
    t = end-start
    label = 'FPS: %.2f ' % (1/t)
    print(label)
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    cv2.imshow('frame',frame)
    # Neu nhan Esc thi thoat
    if cv2.waitKey(1) >= 0:
        break
