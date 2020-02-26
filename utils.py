import math
import numpy as np
import torch


def vectorization(c, char_dict):
    x = np.zeros((len(c), len(char_dict) + 1), dtype=np.bool)
    for i, c_i in enumerate(c):
        if c_i in char_dict:
            x[i, char_dict[c_i]] = 1
        else:
            x[i, 0] = 1
    return x

def get_bounds(data, factor):
    max_x, min_x, max_y, min_y = -math.inf, math.inf, -math.inf, math.inf
    
    abs_x = 0.0
    abs_y = 0.0
    for i in range(len(data)):
        x = float(data[i,0])/factor
        y = float(data[i,1])/factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)
    
    return (min_x, max_x, min_y, max_y)


def sample_to_abs(points, factor=10):
    min_x, max_x, min_y, max_y = get_bounds(points, factor)

    abs_x = 25 - min_x
    abs_y = 25 - min_y

    res = list()
    for i in range(len(points)):
        x = float(points[i, 0]) / factor
        y = float(points[i, 1]) / factor

        abs_x += x
        abs_y += y

        res.append([abs_x, abs_y, points[i, 2]])

    return res

def get_bounds_torch(data, factor):
    data_clone = torch.clone(data)
    for i in range(1, data_clone.shape[0]):
        data_clone[i, :-1] += data_clone[i -1,:-1]
    data_clone[:, :-1] /= factor

    max_x, max_y, _ = torch.max(data_clone, dim=0)[0]
    min_x, min_y, _ = torch.min(data_clone, dim=0)[0]

    return (min_x, max_x, min_y, max_y)

def sample_to_abs_torch(points, factor=10):                                       
    min_x, max_x, min_y, max_y = get_bounds_torch(points, factor)                 
                                                                                  
    abs_x = 25 - min_x
    abs_y = 25 - min_y
     
    points[:, :-1] /= factor 
     
    points[0, 0] += abs_x 
    points[0, 1] += abs_y 
     
    for i in range(1, len(points)): 
        points[i, :-1] += points[i-1,:-1] 
     
    return points 
 
