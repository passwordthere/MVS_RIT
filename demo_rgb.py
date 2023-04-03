# -- coding: utf-8 --

import sys
import threading
import os
import termios
import time

from ctypes import *

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from MvImport.MvCameraControl_header import MV_Image_Bmp, MV_Image_Png
from models import model_dict
from utils import get_predictions

sys.path.append("./MvImport")
from MvCameraControl_class import *

g_bExit = False

device = torch.device("cuda")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def init_model():
    model = model_dict['densenet']
    model = model.to(device)
    filename = 'best_model.pkl'
    # filename = 'dense_net174.pkl'
    model.load_state_dict(torch.load(filename))
    model = model.to(device)
    model.eval()

    return model


def preprocess_frame(frame_gray):
    table = 255.0 * (np.linspace(0, 1, 256) ** 0.8)
    frame_gray = cv2.LUT(np.array(frame_gray), table)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    frame_gray = clahe.apply(np.array(np.uint8(frame_gray)))
    frame_gray = transform(frame_gray)
    frame_gray = torch.unsqueeze(frame_gray, 0)
    return frame_gray


def semantic_segmentation(frame, model):
    # assert len(frame.shape) == 4, 'Frame must be [1,1,H,W]'
    with torch.no_grad():
        data = frame.to(device)
        output = model(data)
        predict = get_predictions(output).squeeze(0)
        pred_img = predict.cpu().numpy() / 3.0
        return pred_img


# 显示图像
def image_show(image):
    # image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = cv2.convertScaleAbs(image, alpha=(255.0))
    cv2.imshow('fgmask', image)
    cv2.waitKey(1) & 0xff


def image_write(image, no):
    cv2.imwrite(f'Semantic_Segmentation_Dataset/test/images/fgmask_{no}.png', image)


# 为线程定义一个函数
def work_thread(cam=0, data_buf=0, nDataSize=0, model=None):
    stDeviceList = MV_FRAME_OUT_INFO_EX()
    memset(byref(stDeviceList), 0, sizeof(stDeviceList))
    while True:
        start_time = time.perf_counter()
        ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nDataSize, stDeviceList, 1000)
        if ret == 0:
            nRGBSize = stDeviceList.nWidth * stDeviceList.nHeight * 4 + 2048
            pDataForSaveImage = (c_ubyte * nRGBSize)()
            stSaveParam = MV_SAVE_IMAGE_PARAM_EX()
            memset(byref(stSaveParam), 0, sizeof(stSaveParam))
            stSaveParam.enImageType = MV_Image_Bmp
            stSaveParam.enPixelType = stDeviceList.enPixelType
            stSaveParam.nBufferSize = stDeviceList.nWidth * stDeviceList.nHeight * 4 + 2048
            stSaveParam.nWidth = stDeviceList.nWidth
            stSaveParam.nHeight = stDeviceList.nHeight
            stSaveParam.pData = data_buf
            stSaveParam.nDataLen = stDeviceList.nFrameLen
            stSaveParam.pImageBuffer = pDataForSaveImage
            stSaveParam.nJpgQuality = 100

            nRet = cam.MV_CC_SaveImageEx2(stSaveParam)
            if nRet == 0:
                img_buff = (c_ubyte * stSaveParam.nImageLen)()
                memmove(byref(img_buff), pDataForSaveImage, stSaveParam.nImageLen)
                img_buffer_numpy = np.frombuffer(img_buff, dtype=np.uint8)
                image = cv2.imdecode(img_buffer_numpy, 1)
                image = image_origin = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = preprocess_frame(image)
                image = semantic_segmentation(image, model)

                image = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                image = cv2.applyColorMap(image, 16)
                image_origin = np.dstack([image_origin, image_origin, image_origin])
                # print(image_origin.shape, image.shape)

                combined = cv2.addWeighted(image_origin, 0.3, image, 0.7, 0)
                image_show(combined)
                # image_show(image)
                # image_show(image_origin)

                print("FPS: ", 1.0 / (time.perf_counter() - start_time))

                # img_buffer_numpy = np.frombuffer(data_buf, dtype=np.uint8)
                # img = cv2.imdecode(img_buffer_numpy, 1)
                # image_show(image=img)
            else:
                print("save failed[0x%x]" % ret)
                sys.exit()
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
        # hThreadHandle = threading.Thread(target=work_thread, args=(cam, data_buf, nPayloadSize))
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
