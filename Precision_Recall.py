import numpy as np
import argparse
import cv2
import time
from math import pow, sqrt
import sys
import os
import csv
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET


# Xu ly tham so dau vao
parser = argparse.ArgumentParser(description='Use MobileNet SSD on Pi for object detection')
parser.add_argument("--path", help="DUong dan", default="C:/Users/Admin/Downloads/Compressed/voc2005_2.tar_2/voc2005_2/test/")
parser.add_argument("--prototxt", default="MobileNetSSD_deploy.prototxt")
parser.add_argument("--weights", default="MobileNetSSD_deploy.caffemodel")
args = parser.parse_args()

# Labels of Network.
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
path_to_images = 'C:/Users/Admin/Downloads/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/'
folder = args.path

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def bb_intersection_over_union(boxA, boxB):
    # Toạ độ hình chữ nhật tương ứng phần giao nhau
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Tính diện tích phần giao nhau
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # Diện tích của predicted và ground-truth bounding box
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Tính IoU = diện tích phần giao nhau chia diện tích phần tổng hợp
    # Diện tích phần hợp = tổng diện tích trừ diện tích phần giao
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Trả về giá trị iou
    return iou
# Load model

def cal_position(detections,i,cols,rows):
    # Lay class_id
    class_id = int(detections[0, 0, i, 1])

    # Tinh toan vi tri cua doi tuong
    xLeftBottom = int(detections[0, 0, i, 3] * cols)
    yLeftBottom = int(detections[0, 0, i, 4] * rows)
    xRightTop = int(detections[0, 0, i, 5] * cols)
    yRightTop = int(detections[0, 0, i, 6] * rows)


    heightFactor = images[i].shape[0] / 300.0
    widthFactor = images[i].shape[1] / 300.0


    xLeftBottom = int(widthFactor * xLeftBottom)
    yLeftBottom = int(heightFactor * yLeftBottom)
    xRightTop = int(widthFactor * xRightTop)
    yRightTop = int(heightFactor * yRightTop)

    return class_id, xLeftBottom, yLeftBottom, xRightTop, yRightTop
def do_detect(frame,net, classNames):
    # Resize anh ve 300x300
    toado = []
    frame_resized = cv2.resize(frame, (300, 300))
    frame_resized = cv2.cvtColor(frame_resized,cv2.COLOR_RGB2BGR)
    # Doc blob va dua vao mang predict
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
            class_id, xLeftBottom, yLeftBottom, xRightTop, yRightTop = cal_position(detections, i, cols, rows)
            # Ve label cua doi tuong
            if class_id == 15:
                global kt
                kt= 1
                # Ve khung hinh chu nhat
                print(dem)
                toado.append([xLeftBottom, yLeftBottom, xRightTop, yRightTop])

    return  toado



net = cv2.dnn.readNetFromCaffe(args.prototxt, args.weights)
# Bat dau doc tu video/webcam
images=load_images(path_to_images +'nguoi/')
print(len(images))
        # Thuc hien detect
dem = 0
kt =0
toadofinal = []
for i in range(len(images)):
    frame = do_detect(images[i],net,classNames)
    if kt ==1:
        dem = dem +1
    kt=0
    toadofinal.append(frame)
    #print(dem)

#print(toadofinal)

path_xml = 'C:/Users/Admin/Downloads/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/xmlnguoi/'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path_xml):
    for file in f:
        if '.xml' in file:
            files.append(os.path.join(r, file))
toadofinal2=[]
for i in files:
    root = ET.parse(i).getroot()
    toado=[]
    #print(root.findall('annotation/object'))
    for type_tag in root.findall('object'):
        if type_tag.find('name').text == "person":
            a=[(int(type_tag.find('bndbox/xmin').text)),int((type_tag.find('bndbox/ymin').text)),int((type_tag.find('bndbox/xmax').text)),int((type_tag.find('bndbox/ymax').text))]
            toado.append(a)
    toadofinal2.append(toado)
#print(toadofinal2)

mau_cua_precision = 0
mau_cua_recall = 0
dem1= 0
list_TP=[]
for i in range(len(images)):
    image = images[i]
    k2 =[]
    k2 =list(range(len(toadofinal2[i])))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    TP =0
    dem_precision = 0
    for j in range(len(toadofinal[i])):
        cv2.rectangle(image, (toadofinal[i][j][0], toadofinal[i][j][1]), (toadofinal[i][j][2], toadofinal[i][j][3]),
                      (0, 255, 0), 3)
        demb=0
        nho = -1
        mau_cua_precision = mau_cua_precision +1
        dem_precision = dem_precision +1
        maxiou = bb_intersection_over_union(toadofinal[i][j],toadofinal2[i][0])
        for k in k2:
            iou = bb_intersection_over_union(toadofinal[i][j],toadofinal2[i][k])
            demb = demb+1
            #print(k2)
            #print(k)
            #print(iou)
            if (iou>0.3) and (iou>=maxiou):
                maxiou=iou
                #print(demb)
                nho = demb
        if (nho>-1):
            TP= TP+1
            dem1=dem1+1
            k2.pop(nho-1)
    list_TP.append([TP, dem_precision-TP])
    for j in range(len(toadofinal2[i])):
        cv2.rectangle(image, (toadofinal2[i][j][0], toadofinal2[i][j][1]), (toadofinal2[i][j][2], toadofinal2[i][j][3]), (0, 0, 255), 2)
        mau_cua_recall = mau_cua_recall +1
    #print(dem1)
    #print(mau_cua_precision)
    #print(mau_cua_recall)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

for i in range(len(list_TP)):
    print(list_TP[i])
print("Precision:", dem1/mau_cua_precision)
print("Recall:", dem1/mau_cua_recall)