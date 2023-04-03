# -- coding: utf-8 --

import sys
import threading
import os
import termios

from ctypes import *

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from models import model_dict
from utils import get_predictions

sys.path.append("./MvImport")
from MvCameraControl_class import *

g_bExit = False

device = torch.device("cuda")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])


def init_model():
    model = model_dict['densenet']
    model = model.to(device)
    filename = 'best_model.pkl'
    if not os.path.exists(filename):
        print("model path not found !!!")
        exit(1)

    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()

    return model


def preprocess_frame(frame):
    table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
    frame = cv2.LUT(np.array(frame), table)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    frame = clahe.apply(np.array(np.uint8(frame)))
    frame = transform(frame)
    # frame = (frame - frame.mean())/frame.std()
    # frame = torch.from_numpy(frame).unsqueeze(0).to(torch.float32)
    frame = torch.unsqueeze(frame, 0)
    return frame


def semantic_segmentation(frame, model):
    # assert len(frame.shape) == 4, 'Frame must be [1,1,H,W]'
    with torch.no_grad():
        data = frame.to(device)
        output = model(data)
        predict = get_predictions(output).squeeze(0)
        # predict = predict.squeeze().numpy()
        pred_img = predict.cpu().numpy() / 3.0
        return pred_img



# 显示图像
def image_show(image):
    image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = cv2.convertScaleAbs(image, alpha=(255.0))
    cv2.imshow('fgmask', image)
    k = cv2.waitKey(1) & 0xff


def image_write(image, no):
    cv2.imwrite(f'Semantic_Segmentation_Dataset/test/images/fgmask_{no}.png', image)


# 为线程定义一个函数
def work_thread(cam=0, data_buf=0, nDataSize=0, model=None):
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nDataSize, stFrameInfo, 5000)
        if ret == 0:
            # print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))

            image = np.asarray(data_buf)
            image = image.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)

            # image = preprocess_frame(image)
            # image = semantic_segmentation(image, model)
            print(image.shape)
            image_show(image=image)

            # plt.imshow(image)
            # plt.imsave('Semantic_Segmentation_Dataset/test/output/{}.jpg', pred_img)
            # plt.show()

            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # image_show(image=image)
            # image_write(image=image, no=stFrameInfo.nFrameNum)
        else:
            print("no data[0x%x]" % ret)
        if g_bExit == True:
            break


def press_any_key_exit():
    fd = sys.stdin.fileno()
    old_ttyinfo = termios.tcgetattr(fd)
    new_ttyinfo = old_ttyinfo[:]
    new_ttyinfo[3] &= ~termios.ICANON
    new_ttyinfo[3] &= ~termios.ECHO
    # sys.stdout.write(msg)
    # sys.stdout.flush()
    termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
    try:
        os.read(fd, 7)
    except:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)


if __name__ == "__main__":
    model = init_model()

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄| en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:获取数据包大小 | en:Get payload size
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    nPayloadSize = stParam.nCurValue

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    data_buf = (c_ubyte * nPayloadSize)()

    try:
        hThreadHandle = threading.Thread(target=work_thread, args=(cam, data_buf, nPayloadSize, model))
        hThreadHandle.start()
    except:
        print("error: unable to start thread")

    print("press a key to stop grabbing.")
    press_any_key_exit()

    g_bExit = True
    hThreadHandle.join()

    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    del data_buf
