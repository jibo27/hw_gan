import os
import numpy as np
import cv2
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn

from discriminator import Discriminator
from generator import Generator
from psf import extract_psf_torch, imshow, extract_psf
from utils import sample_to_abs_torch, vectorization, sample_to_abs
from dataloader import IAMDataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad_imgs(imgs, sort):
    max_width = max([img.shape[-1] for img in imgs])
    if sort == False:
        indices = list(range(len(imgs)))
    else:
        data = sorted(enumerate(imgs), key=lambda x: x[1].shape[-1], reverse=True)
        indices, imgs = zip(*data)
    
    widths = [img.shape[-1] for img in imgs]

    padded_imgs = list()

    for index, img in enumerate(imgs):
        padded_imgs.append(F.pad(img, (0, max_width - img.shape[-1]), 'constant', 0))
    imgs = torch.stack(padded_imgs, 0)
    
    return imgs, widths, list(indices)


def get_fake_data(generator, args, texts=None):
    if args.mode == 'prediction':
        points_list = generator.sample(args.batch_size, args.T)
    elif args.mode == 'synthesis':
        vec = np.zeros([args.batch_size, args.U, args.c_dimension])
        char_to_indices = dict((c, i + 1) for i, c in enumerate(args.chars))
        for i in range(args.batch_size):
            vec[i] = vectorization(texts[i], char_to_indices)
        points_list = generator.sample(args.batch_size, args.T, vec)
    else:
        raise ValueError

    fake_psfs = list()
    for points in points_list:
        fake_psf = extract_psf_torch(sample_to_abs_torch(points), scalar_x=20, flipud=True, S=args.S, S_scalar=args.S_scalar)
        fake_psf = fake_psf.permute(2, 0, 1)
        fake_psfs.append(fake_psf)

    fake_psf, fake_widths, _ = pad_imgs(fake_psfs, sort=False)

    for i in range(args.batch_size):
        cv2.imwrite('outputs/fake_psf_%d.png'%i, imshow((fake_psf.cpu().detach().numpy())[i][0], False))

    return fake_psf, fake_widths




def pre_d(discriminator, generator, train_loader, num_steps):
    loss_fn = nn.BCELoss()
    generator.eval()
    for step in range(num_steps):
        with torch.no_grad():
            if args.mode == 'prediction':
                fake_psf, fake_widths = get_fake_data(generator, args)
                x, _1, _2, _3 = train_loader.random_batch()
                real_psf = list()
                for i in range(args.batch_size):
                    psf = extract_psf(sample_to_abs(x[i]), scalar_x=20, flipud=True, S=None)
                    psf = psf.transpose(2, 0, 1)
                    real_psf.append(torch.Tensor(psf).to(device))

                real_psf, real_widths, _ = pad_imgs(real_psf, sort=True) 

                for i in range(args.batch_size):
                    cv2.imwrite('outputs/real_psf_%d.png'%i, imshow((real_psf.cpu().detach().numpy())[i][0], False))

            elif args.mode == 'synthesis':
                x, y, c_vec, texts = train_loader.random_batch()
                real_psf = list()
                for i in range(args.batch_size):
                    psf = extract_psf(sample_to_abs(x[i]), scalar_x=20, flipud=True, S=None)
                    psf = psf.transpose(2, 0, 1)
                    real_psf.append(torch.Tensor(psf).to(device))

                real_psf, real_widths, indices = pad_imgs(real_psf, sort=True) 
                texts = np.array(texts)
                texts = list(texts[indices])

                fake_psf, fake_widths = get_fake_data(generator, args, texts)
                
                for i in range(args.batch_size):
                    cv2.imwrite('outputs/real_psf_%d.png'%i, imshow((real_psf.cpu().detach().numpy())[i][0], False))
            else:
                raise ValueError

        D_fake = discriminator.predict(fake_psf, fake_widths)
        D_real = discriminator.predict(real_psf, real_widths)

        true_labels = torch.Tensor([1] * args.batch_size).to(device)
        fake_labels = torch.Tensor([0] * args.batch_size).to(device)
        D_loss = loss_fn(D_fake, fake_labels) + loss_fn(D_real, true_labels)

        discriminator.optimizer.zero_grad()
        D_loss.backward()
        discriminator.optimizer.step()

        if step % 20 == 0:
            print('Step: %d | Loss: %f'%(step, D_loss))

            d_path = 'models/discriminator_%d.pt'%(step)
            torch.save(discriminator.state_dict(), d_path)

    generator.train()

def pre_g(generator, train_loader, num_epochs, mode):
    for epoch in range(num_epochs):
        print("Epoch: %d" % epoch)
        train_loader.reset_batch_pointer()
        for step in range(train_loader.num_batches):
            x, y, c_vec, c = train_loader.next_batch()
            if mode == 'prediction':
                generator.fit(x, y)
            elif mode == 'synthesis':
                generator.fit(x, y, c_vec)
            else:
                raise ValueError
            if step % 20 == 0:
                print('Step: %d | loss: %.6f'%(step, generator.loss.cpu().item()))

        if epoch != 0 and (epoch + 1) % 5 == 0:
            save_path = 'models/%s_generator_%d.pt'%(mode, epoch + 1)
            torch.save(generator.state_dict(), save_path)



def ad_train(args, generator, discriminator, train_loader, num_steps):
    loss_fn = nn.BCELoss()
    g_scheduler = torch.optim.lr_scheduler.ExponentialLR(generator.optimizer, gamma=0.95)
    d_scheduler = torch.optim.lr_scheduler.ExponentialLR(discriminator.optimizer, gamma=0.95)
    for step in range(num_steps):
        if args.mode == 'prediction':
            fake_psf, fake_widths = get_fake_data(generator, args)
            x, _1, _2, _3 = train_loader.random_batch()
            real_psf = list()
            for i in range(args.batch_size):
                psf = extract_psf(sample_to_abs(x[i]), scalar_x=20, flipud=True, S=None)
                psf = psf.transpose(2, 0, 1)
                real_psf.append(torch.Tensor(psf).to(device))

            real_psf, real_widths, _ = pad_imgs(real_psf, sort=True) 

            for i in range(args.batch_size):
                cv2.imwrite('outputs/real_psf_%d.png'%i, imshow((real_psf.cpu().detach().numpy())[i][0], False))

        elif args.mode == 'synthesis':
            x, y, c_vec, texts = train_loader.random_batch()
            real_psf = list()
            for i in range(args.batch_size):
                psf = extract_psf(sample_to_abs(x[i]), scalar_x=20, flipud=True, S=None)
                psf = psf.transpose(2, 0, 1)
                real_psf.append(torch.Tensor(psf).to(device))

            real_psf, real_widths, indices = pad_imgs(real_psf, sort=True) 
            texts = np.array(texts)
            texts = list(texts[indices])

            fake_psf, fake_widths = get_fake_data(generator, args, texts)
            
            for i in range(args.batch_size):
                cv2.imwrite('outputs/real_psf_%d.png'%i, imshow((real_psf.cpu().detach().numpy())[i][0], False))
        else:
            raise ValueError

        D_fake = discriminator.predict(fake_psf, fake_widths)
        D_real = discriminator.predict(real_psf, real_widths)

        D_loss = 0.0
        if bool(args.ad_d_use_labels) is True:
            true_labels = torch.Tensor([1] * args.batch_size).to(device)
            fake_labels = torch.Tensor([0] * args.batch_size).to(device)
            D_loss += loss_fn(D_fake, fake_labels) + loss_fn(D_real, true_labels)


        D_loss += - torch.mean(torch.log(D_real)) - torch.mean(torch.log(1 - D_fake))

        discriminator.optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        discriminator.optimizer.step()

        if bool(args.ad_label_smoothing) is True:
            true_labels = torch.clamp(torch.rand(args.batch_size).to(device) * 0.4 + 0.7, 0, 1)

        G_loss = loss_fn(D_fake.squeeze(), true_labels)

        generator.optimizer.zero_grad()
        G_loss.backward()
        if bool(args.ad_g_clip_norm) is True: # clip gradients
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        generator.optimizer.step()


        if step != 0 and (step + 1) % 10 == 0:
            d_path = 'models/ad_discriminator_%d.pt'%(step + 1)
            torch.save(discriminator.state_dict(), d_path)

            g_path = 'models/ad_%s_generator_%d.pt'%(args.mode, step + 1)
            torch.save(generator.state_dict(), g_path)

            # adjust learning rate
            g_scheduler.step()
            d_scheduler.step()


def main(args):
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # -------------- dataset ----------------------------
    g_train_loader = IAMDataLoader(args.batch_size, args.T, args.data_scale, chars=args.chars, points_per_char=args.points_per_char)
    print('number of batches:', g_train_loader.num_batches)
    args.c_dimension = len(g_train_loader.chars) + 1

    args.U = g_train_loader.max_U

    # -------------- pretrain generator ----------------------------
    generator = Generator(num_gaussians=args.M, mode=args.mode, c_dimension=args.c_dimension, K=args.K, U=args.U, batch_size=args.batch_size, T=args.T, bias=args.b, sample_random=args.sample_random, learning_rate=args.g_learning_rate).to(device)
    generator = generator.train()

    if args.g_path and os.path.exists(args.g_path):
        print('Start loading generator: %s'%(args.g_path))
        generator.load_state_dict(torch.load(args.g_path))
    else:
        print('Start pre-training generator:')
        pre_g(generator, g_train_loader, num_epochs=40, mode=args.mode)


    # -------------- pretrain discriminator ----------------------------
    if args.batch_size > 16: # Do not set batch_size too large
        generator.batch_size = 16
        args.batch_size = 16
    discriminator = Discriminator(learning_rate=args.d_learning_rate, weight_decay=args.d_weight_decay).to(device)
    discriminator = discriminator.train()

    if args.d_path and os.path.exists(args.d_path):
        print('Start loading discriminator: %s'%(args.d_path))
        discriminator.load_state_dict(torch.load(args.d_path))
    else:
        print('Start pre-training discriminator:')
        pre_d(discriminator, generator, g_train_loader, num_steps=200)


    generator.set_learning_rate(args.ad_g_learning_rate)

    print('Start training discriminator:')
    ad_train(args, generator, discriminator, g_train_loader, num_steps=100)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['prediction', 'synthesis'], required=True)
    parser.add_argument('--text', type=str, default=None) 

    parser.add_argument('--S', type=float, default=1, help='param for interpolation')
    parser.add_argument('--S_scalar', type=float, default=1, help='scale S')

    parser.add_argument('--d_learning_rate', type=float, default=1e-3)
    parser.add_argument('--d_weight_decay', type=float, default=1e-5)
    parser.add_argument('--d_path', type=str, default=None, help='path to discrimiator model')

    parser.add_argument('--g_learning_rate', type=float, default=1e-3)
    parser.add_argument('--ad_g_learning_rate', type=float, default=1e-5)
    parser.add_argument('--g_path', type=str, default=None)

    parser.add_argument('--ad_g_clip_norm', type=int, default=1)
    parser.add_argument('--ad_label_smoothing', type=int, default=1)
    parser.add_argument('--ad_d_use_labels', type=int, default=1)

    parser.add_argument('--sample_random', type=bool, default=False, help='apply random-control sample or not')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--chars', type=str, default='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ')
    parser.add_argument('--T', type=int, default=300, help='RNN sequence length')
    parser.add_argument('--points_per_char', type=int, default=25, help='points per char (appr.)')
    parser.add_argument('--num_layers', type=int, default=2, help='num of RNN stack layers')
    parser.add_argument('--M', type=int, default=20, help='num of mixture bivariate gaussian')
    parser.add_argument('--K', type=int, default=5, help='num of mixture bivariate gaussian (for synthesis)')
    parser.add_argument('--data_scale', type=float, default=20, help='factor to scale raw data down by')
    parser.add_argument('--b', type=float, default=3.0, help='biased sampling used in sampling')
    parser.add_argument('--train_b', type=float, default=3.0, help='biased sampling used in training')
    args = parser.parse_args()


    main(args)
