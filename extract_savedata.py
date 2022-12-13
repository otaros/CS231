
import os
import pickle
import cv2 as cv
import numpy as np
import pandas as pd


def extract_features(path='data'):
    data = [[[0, 255, 0]], [1]]
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            if file.endswith('.jpg'):
                pathFile = os.path.join(path, folder, file)
                image = cv.imread(pathFile, 3)
                features = image.reshape(-1, 3)
                label = file.split('-')[1][0]
                data[0] = np.append(data[0], features, axis=0)
                data[1] += [label for _ in range(features.shape[0])]

    coefficients = np.array([[1] for _ in range(len(data[0]))])
    data[0] = np.append(data[0], coefficients, axis=1)

    return data


def save_data(data, name):
    with open(name + '.pickle', 'wb') as f:
        pickle.dump(data, f)


data = extract_features('data')
df = pd.DataFrame()
df['pixel'] = list(map(str, data[0]))
df['labels'] = list(map(int, data[1]))
df = df.drop_duplicates(subset='pixel')
df.to_csv('data.csv', index=False)
save_data(data, 'data')
