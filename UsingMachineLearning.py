import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle
import os
import time
import pandas as pd
import sklearn


def loadmodel(name='SVC'):
    with open(f'./model/{name}.pickle', 'rb') as f:
        model = pickle.load(f)
    return model


def removeGreen(img, model, bg):
    if img.shape != bg.shape:
        bg = cv.resize(bg, (img.shape[1], img.shape[0]),
                       fx=0, fy=0, interpolation=cv.INTER_CUBIC)

    mask = model.predict(img.reshape(-1, 3))
    mask = mask.reshape(img.shape[0], img.shape[1])
    img[mask == '1'] = bg[mask == '1']

    return img


def videoProcesing(model, bg, name='vid-1', model_name='') -> None:
    cap = cv.VideoCapture(f'./video/{name}.mp4')
    # fourcc = cv.VideoWriter_fourcc('h', '2', '6', '4')
    # out_name = name.split('/')[1].split('.')[0]
    """ out = cv.VideoWriter(
        f'output_vid/{model_name}_{name}.mp4', fourcc, 30, (640, 480)) """
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv.GaussianBlur(frame, (0, 0), sigmaX=1, sigmaY=1, borderType=cv.BORDER_DEFAULT)
            resized_frame = cv.resize(
                frame, (720, 560), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
            processed_frame = removeGreen(resized_frame, model, bg)
            # out.write(processed_frame)
            cv.imshow(f'{model_name}', processed_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    # out.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    fg_num = input("Enter video number:")
    bg_num = input("Enter background number:")
    # model_name = input("Enter the name of the model: ")
    model_name = 'SGDClassifier'

    model = loadmodel(model_name)
    bg = cv.imread(f'background/bg-{bg_num}.jpg', 1)
    videoProcesing(model, bg, f'vid-{fg_num}', model_name)
