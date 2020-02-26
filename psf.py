import numpy as np
import copy
import pickle
import math
import cv2
import torch

from interpolation import interpolation, interpolation_torch

def imshow(image, visiable):
    new_image = (image + 1.0) / 2.0 * 255
    new_image[new_image == 127.5 ] = 127.5

    new_image = np.array(new_image, np.uint8)

    if visiable:
        cv2.imshow('winname', new_image)
        cv2.waitKey(0)

    return new_image


# ------------------------------------ numpy -------------------------------------
def _resample(points, S=None, S_scalar=1):
    points_resampled = list()
    prev_index = 0 
    for index, point in enumerate(points):
        if point[2] == 1:
            if prev_index != index:
                points_resampled_tmp = interpolation(points[prev_index: index + 1], S, S_scalar)
                points_resampled.extend([[point[0], point[1], 1 if index + 1 == len(points_resampled_tmp) else 0] for index, point in enumerate(points_resampled_tmp)])
            prev_index = index + 1
    return points_resampled


def _scale(points, height=128, scalar_x=0.2):
    max_x, min_x, max_y, min_y = -math.inf, math.inf, -math.inf, math.inf

    max_x = max(max_x, max([point[0] for point in points]))
    min_x = min(min_x, min([point[0] for point in points]))
    max_y = max(max_y, max([point[1] for point in points]))
    min_y = min(min_y, min([point[1] for point in points]))

    margin = 2
    scalar_y = float(height - margin * 2) / float(max_y - min_y)
    scalar_x = scalar_x

    width = int(((max_x - min_x)*scalar_x + margin * 2))

    shape = (height, width, 3)
    
    
    points_scaled = list()
    for point in points:
        points_scaled.append(((point[0] - min_x) * scalar_x, (point[1] - min_y) * scalar_y, point[2]))

    return points_scaled, shape
    
def extract_psf(points, height=128, scalar_x=0.2, threshold=None, flipud=False, S=None):
    points, shape = _scale(points, height, scalar_x) 

    points = _resample(points, S=S)

    if threshold is not None:
        if shape[0] < threshold[0] or shape[1] < threshold[1]:
            return None

    padding = 2
    channel = 7
    
    psf = np.zeros((int(shape[0]), int(shape[1]), int(channel)), np.float) 

    prev_index = 0
    for index, point in enumerate(points):
        if point[2] == 0:
            continue
        if prev_index == index:
            prev_index + 1
            continue

        stroke = points[prev_index: index + 1]
        prev_index = index + 1

        point_features = np.zeros((len(stroke), int(channel)))

        for i, p in enumerate(stroke):
            features = np.zeros(int(channel), np.float)
            features[0] = 1.0

            if i != len(stroke) - 1:
                features[1] = stroke[i + 1][0] - stroke[i][0]
                features[2] = stroke[i + 1][1] - stroke[i][1]
                features[3] = 0.5 * (features[1] ** 2)
                features[4] = 0.5 * (features[1] * features[2])
                features[5] = 0.5 * (features[2] * features[1])
                features[6] = 0.5 * (features[2] ** 2)                    
            point_features[i] = copy.deepcopy(features)

        for c in range(channel):
            max_value = np.amax(point_features[:, c])
            min_value = np.amin(point_features[:, c])

            if max_value != min_value:
                point_features[:, c] = 2.0 * (point_features[:, c] - min_value) / (max_value - min_value) - 1.0
            elif max_value == min_value and max_value != 0:
                point_features[:, c] /= max_value

        for i, point in enumerate(stroke):
            if flipud == False:
                psf[height - int(point[1] + padding)][int(point[0] + padding)][:] = point_features[i]
            else:
                psf[int(point[1] + padding)][int(point[0] + padding)][:] = point_features[i]

    return psf


# ------------------------------------ torch -------------------------------------
def _resample_torch(points, S=None, S_scalar=1):
    points_resampled = list()
    prev_index = 0 
    for index, point in enumerate(points):
        if point[2] == 1:
            if prev_index != index:
                points_resampled_tmp = interpolation_torch(points[prev_index: index + 1], S, S_scalar)
                points_resampled.extend(points_resampled_tmp)
                points_resampled[-1][-1] = 1
                
            prev_index = index + 1
    return points_resampled



def _scale_torch(points, height=128, scalar_x=0.2):
    max_x, max_y, _ = torch.max(points, dim=0)[0]   
    min_x, min_y, _ = torch.min(points, dim=0)[0]   

    margin = 2
    scalar_y = float(height - margin * 2) / float(max_y - min_y)
    scalar_x = scalar_x

    width = int(((max_x - min_x)*scalar_x + margin * 2))

    shape = (height, width, 3)
    
    points[:, 0] = (points[:, 0] - min_x) * scalar_x
    points[:, 1] = (points[:, 1] - min_y) * scalar_y
    return points, shape
    



def extract_psf_torch(points, height=128, scalar_x=0.2, threshold=None, flipud=False, S=None, S_scalar=1, resample_flag=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    points, shape = _scale_torch(points, height, scalar_x) # scale first, then sampling
    
    if resample_flag == True:
        points = _resample_torch(points, S=S, S_scalar=S_scalar)

    if threshold is not None:
        if shape[0] < threshold[0] or shape[1] < threshold[1]:
            return None


    padding = 2
    channel = 7
    
    psf = torch.zeros((int(shape[0]), int(shape[1]), int(channel)), dtype=torch.float).to(device)

    prev_index = 0
    for index, point in enumerate(points):
        if point[2] == 0:
            continue
        if prev_index == index:
            prev_index + 1
            continue

        stroke = points[prev_index: index + 1]
        prev_index = index + 1

        point_features = torch.zeros((len(stroke), int(channel))).to(device)

        for i, p in enumerate(stroke):
            features = torch.zeros(int(channel), dtype=torch.float).to(device)
            
            features[0] = 1.0

            if i != len(stroke) - 1:
                a = stroke[i + 1][0] - stroke[i][0]
                b = stroke[i + 1][1] - stroke[i][1]
                features[1] = a
                features[2] = b
                features[3] = 0.5 * (a ** 2)
                features[4] = 0.5 * (a * b)
                features[5] = 0.5 * (b * a)
                features[6] = 0.5 * (b ** 2)                    
            point_features[i] = features

        point_features_normalized = torch.zeros((len(stroke), int(channel))).to(device)

        for c in range(channel):
            max_value = torch.max(point_features[:, c])
            min_value = torch.min(point_features[:, c])

            if max_value != min_value:
                point_features_normalized[:, c] = 2.0 * (point_features[:, c] - min_value) / (max_value - min_value) - 1.0
            elif max_value == min_value and max_value != 0:
                point_features_normalized[:, c] = point_features[:, c] / max_value

        for i, point in enumerate(stroke):
            if flipud == False:
                psf[height - int(point[1] + padding)][int(point[0] + padding)][:] = point_features_normalized[i]
            else:
                psf[int(point[1] + padding)][int(point[0] + padding)][:] = point_features_normalized[i]
    return psf


