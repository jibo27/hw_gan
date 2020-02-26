import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import argparse

from generator import Generator
from psf import extract_psf, imshow, extract_psf_torch
from utils import sample_to_abs_torch, vectorization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--num_samples', type=int, default=8, help='number of handwritten texts to be generated')

parser.add_argument('--mode', type=str, choices=['prediction', 'synthesis'], required=True)
parser.add_argument('--text', type=str, default=None, help='text to be generated') 

parser.add_argument('--g_learning_rate', type=float, default=1e-4, help='learning rate for generator')
parser.add_argument('--g_path', type=str, default=None, help='path to generator model')

parser.add_argument('--sample_random', type=bool, default=False, help='apply random-control sample or not')
parser.add_argument('--batch_size', type=int, default=8, help='minibatch size')
parser.add_argument('--chars', type=str, default='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ')
parser.add_argument('--T', type=int, default=300, help='RNN sequence length')
parser.add_argument('--points_per_char', type=int, default=25, help='points per char (appr.)')
parser.add_argument('--M', type=int, default=20, help='num of mixture bivariate gaussian')
parser.add_argument('--K', type=int, default=5, help='num of mixture bivariate gaussian (for synthesis)')
parser.add_argument('--b', type=float, default=3.0, help='biased sampling used in sampling')
args = parser.parse_args()


args.c_dimension = len(args.chars) + 1
char_to_indices = dict((c, i + 1) for i, c in enumerate(args.chars))

if args.mode == 'prediction':
    args.U = 20
 
elif args.mode == 'synthesis':
    assert args.text is not None
    os.makedirs(os.path.join('outputs', args.text.strip()), exist_ok=True)
    s = args.text
    s = '  ' + s + '  '
    args.U = len(s)
    args.batch_size = 1
    vec = vectorization(s, char_to_indices)
else:
    raise ValueError


args.T = 1

generator = Generator(num_gaussians=args.M, mode=args.mode, c_dimension=args.c_dimension, K=args.K, U=args.U, batch_size=args.batch_size, T=args.T, bias=args.b, sample_random=args.sample_random, learning_rate=args.g_learning_rate).to(device)


def convert_to_img(img, threshold=0.5):
    res = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            res[i, j] = 255 if img[i, j] < threshold else 0
    return res


if args.g_path and os.path.exists(args.g_path):
    generator.load_state_dict(torch.load(args.g_path))
    generator = generator.eval()

    if args.mode == 'prediction':
        batch_size = args.num_samples
        with torch.no_grad():
            points_list = generator.sample(batch_size, 800)
        for index, points in enumerate(points_list):
            fake_psf = extract_psf_torch(sample_to_abs_torch(points, factor=0.1), flipud=True, S_scalar=0.1)
            fake_psf = fake_psf.cpu().detach().numpy()

            dirname = os.path.join('outputs', os.path.splitext(args.g_path)[0].split('/')[-1])
            os.makedirs(dirname, exist_ok=True)
            filename = os.path.join(dirname, 'b%d_%d.png'%(args.b, index))
            print('Writing image to %s'%(filename))

            img = convert_to_img(fake_psf[:, :, 0])
            cv2.imwrite(filename, img)

    elif args.mode == 'synthesis':
        for index in range(args.num_samples):
            with torch.no_grad():
                points_list = generator.sample(args.batch_size, args.U * args.points_per_char, s=vec)
            points = points_list[0]
            fake_psf = extract_psf_torch(sample_to_abs_torch(points, factor=0.1), flipud=True)
            fake_psf = fake_psf.cpu().detach().numpy()
            dirname = os.path.join('outputs', args.text.strip(), os.path.splitext(args.g_path)[0].split('/')[-1])
            os.makedirs(dirname, exist_ok=True)
            filename = os.path.join(dirname, 'b%d_%d.png'%(args.b, index))
            print('Writing image to %s'%(filename))

            img = convert_to_img(fake_psf[:, :, 0])
            cv2.imwrite(filename, img)
    else:
        raise ValueError

