import math
import copy
import torch

# ------------------------------- numpy ---------------------------------
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_resample_spacing(points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    
    top_left_x = min(xs)
    top_left_y = min(ys)
    top_left = [top_left_x, top_left_y]
    
    bottom_right_x = max(xs)
    bottom_right_y = max(ys)
    bottom_right = [bottom_right_x, bottom_right_y]

    diagonal = distance(top_left, bottom_right)

    S = diagonal / 300.0

    return S

def get_resample_points(points, S):
    D = 0
    resampled = []
    resampled.append(points[0])
    point = [None, None]

    i = 1
    while i < len(points):
        d = distance(points[i - 1], points[i])
        if D + d >= S and d != 0:
            px = points[i - 1][0] + ((S - D) / d) * (points[i][0] - points[i - 1][0])
            py = points[i - 1][1] + ((S - D) / d) * (points[i][1] - points[i - 1][1])
            point = [px, py]
            resampled.append(point)
            points.insert(i, point)
            D = 0
        else:
            D = D + d
        i += 1
    return resampled


def interpolation(points, S=None, S_scalar=1):
    if S is None:
        S = get_resample_spacing(points)
    S *= S_scalar
    resampled = get_resample_points(points, S)
    return resampled


# ------------------------------- torch ---------------------------------
def get_resample_spacing_torch(points):
    bottom_right = torch.max(points, dim=0)[0][:-1]
    top_left = torch.min(points, dim=0)[0][:-1]

    diagonal = torch.norm(top_left - bottom_right)

    S = diagonal / 300.0

    return S


def get_resample_points_torch(points, S):
    D = 0
    resampled = []
    resampled.append(points[0])

    points_l = [points[i] for i in range(points.shape[0])]

    i = 1
    while i < len(points_l):
        d = torch.norm(points_l[i - 1][:-1] - points_l[i][ :-1])
        if D + d >= S and d != 0:
            point = torch.zeros(3).to(points.device)
            point[:-1] = points_l[i - 1][:-1] + ((S - D) / d) * (points_l[i][:-1] - points_l[i - 1][:-1])
        
            resampled.append(point)
            points_l.insert(i, point)
            D = 0
        else:
            D = D + d
        i += 1
    return resampled

def interpolation_torch(points, S=None, S_scalar=1):
    if S is None:
        S = get_resample_spacing_torch(points)
    S *= S_scalar
    resampled = get_resample_points_torch(points, S)
    return resampled
