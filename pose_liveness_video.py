import os

from torch.autograd import Variable

import hopenet
import torchvision
import torch
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F

up_head = 1
down_head = 1
left_head = 1
right_head = 1

def main():
    global up_head,down_head,left_head,right_head
    model = load_model()

    transformations = transforms.Compose([transforms.Resize(224),
                                          transforms.CenterCrop(224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    cap = cv2.VideoCapture(0)
    while True:
        _,img = cap.read()
        cv2.rectangle(img,(200,120),(440,360),(0,255,0),2)
        img[:480,:200,:] = np.full((480,200,3),0)
        img[:480,440:,:] = np.full((480,200,3),0)
        img[:120,200:440,:] = np.full((120,240,3),0)
        img[360:,200:440,:] = np.full((120,240,3),0)

        roi_img = img[120:360,200:440]
        yaw,pitch = pred_yaw(model, roi_img, transformations)
        print(up_head,down_head,left_head,right_head)
        if up_head:

            print("抬头检测。。。。。。。")
            cv2.putText(img,"Please raise your head slowly!",
                 (200,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,[0, 255, 0], 2)
            cv2.imshow("Test",img)
            if cv2.waitKey(10) == 27 or pitch.data.cpu().numpy().tolist()>=40:
                up_head = 0

        if down_head and up_head == 0:
            print("低头检测。。。。。。。")
            cv2.putText(img,"Please down your head slowly!",
                  (200,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,[0, 255, 0], 2)
            cv2.imshow("Test",img)
            if cv2.waitKey(10) == 27 or pitch.data.cpu().numpy().tolist()<=-30:
                 down_head = 0


        if left_head and up_head == 0 and down_head == 0:
            print("向左扭脸检测。。。。。。")
            cv2.putText(img,"Please move your head to left slowly!",
                (200,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,[0, 255, 0], 2)
            cv2.imshow("Test",img)
            if cv2.waitKey(10) == 27 or yaw.data.cpu().numpy().tolist()<=-40:
                left_head = 0


        if right_head and up_head==0 and down_head==0 and left_head==0:
            print("向右扭脸检测。。。。。。")
            cv2.putText(img,"Please move your head to right slowly!",
                   (200,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,[0, 255, 0], 2)
            cv2.imshow("Test",img)
            if cv2.waitKey(10) == 27 or yaw.data.cpu().numpy().tolist() >= 40:
                right_head = 0

        if right_head==0 and up_head==0 and down_head==0 and left_head==0:
            cv2.putText(img,"Real Person!",
                    (200,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,[0, 255, 0], 2)
            cv2.imshow("Test",img)
            if cv2.waitKey(10) == 27:
                cv2.destroyAllWindows()
                break


       









def read_img(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8),
                       cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_model():
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    model.load_state_dict(torch.load(r'hopenet_robust_alpha1.pkl'))
    model.cuda(0)
    model.eval()
    return model


def pred_yaw(model, img, transformations=None):
    if transformations is not None:
        img = transformations(Image.fromarray(img))
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(0)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = Variable(img).cuda(0)

    yaw, pitch, roll = model(img)
    yaw_pred = F.softmax(yaw, dim=1)
    yaw_pred = torch.sum(yaw_pred.data[0] * idx_tensor) * 3 - 99
    pitch_pred = F.softmax(pitch, dim=1)
    pitch_pred = torch.sum(pitch_pred.data[0] * idx_tensor) * 3 - 99
    return yaw_pred,pitch_pred





if __name__ == '__main__':
    main()
