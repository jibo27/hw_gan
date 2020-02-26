import numpy as np
import torch
from torch.distributions import MultivariateNormal
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(torch.nn.Module):
    def __init__(self, num_gaussians, mode, c_dimension, K, U, batch_size, T, bias, sample_random, learning_rate):
        super(Generator, self).__init__()

        self.num_gaussians = num_gaussians
        self.output_size = 1 + self.num_gaussians * 6
        self.mode = mode

        # -------------------- Prediction Network --------------------
        self.lstm1 = torch.nn.LSTM(input_size=3, hidden_size=512, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=512 + self.lstm1.input_size, hidden_size=256, batch_first=True)
        self.lstm3 = torch.nn.LSTM(input_size=256 + self.lstm1.input_size, hidden_size=512, batch_first=True)
        self.fc_output = torch.nn.Linear(self.lstm1.hidden_size + self.lstm2.hidden_size + self.lstm3.hidden_size, self.output_size)

        # -------------------- Synthesis Network ----------------------
        self.batch_size = batch_size
        self.K = K
        self.T = T
        self.c_dimension = c_dimension
        self.rnn_cell1 = nn.LSTMCell(input_size=3 + self.c_dimension, hidden_size=512)
        self.rnn_cell2 = nn.LSTMCell(input_size=3 + self.c_dimension + 512, hidden_size=512)
        self.h2k = nn.Linear(512, self.K * 3)
        self.u = torch.arange(U).float().unsqueeze(0).repeat(self.K, 1)
        self.u = self.u.unsqueeze(0).repeat(self.batch_size, 1, 1).to(device)
        self.fc_output_syn = nn.Linear(512, self.output_size)

        self.seq_length = self.T

        self.bias = bias
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)

        self.sample_random = sample_random

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def _bivariate_gaussian(self, x1, x2, mu1, mu2, sigma1, sigma2, rho):
        z = torch.pow((x1 - mu1) / sigma1, 2) + torch.pow((x2 - mu2) / sigma2, 2) \
            - 2 * rho * (x1 - mu1) * (x2 - mu2) / (sigma1 * sigma2)
        return torch.exp(-z / (2 * (1 - torch.pow(rho, 2)))) / \
               (2 * np.pi * sigma1 * sigma2 * torch.sqrt(1 - torch.pow(rho, 2)))
   

    def fit(self, x, y, c_vec=None):
        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)


        if self.mode == 'prediction':
            outputs1, _ = self.lstm1(x, None)
            outputs2, _ = self.lstm2(torch.cat([x, outputs1], dim=2), None)
            outputs3, _ = self.lstm3(torch.cat([x, outputs2], dim=2), None)

            outputs = self.fc_output(torch.cat([outputs1, outputs2, outputs3], dim=2).reshape(-1, self.fc_output.in_features))
        elif self.mode == 'synthesis':
            w = torch.zeros(self.batch_size, self.c_dimension).to(device)
            kappa_prev = torch.zeros([self.batch_size, self.K, 1]).to(device)
            cell1_state, cell2_state = None, None

            output_list = torch.zeros(self.batch_size, self.T, 512).to(device)
            for t in range(self.T):
                cell1_state = self.rnn_cell1(torch.cat([x[:,t,:], w], 1), cell1_state)
                k_gaussian = self.h2k(cell1_state[0])

                alpha_hat, beta_hat, kappa_hat = torch.split(k_gaussian, self.K, dim=1)

                alpha = torch.exp(alpha_hat).unsqueeze(2)
                beta = torch.exp(beta_hat).unsqueeze(2)

                self.kappa = kappa_prev + torch.exp(kappa_hat).unsqueeze(2) # (B, K, 1)
                kappa_prev = self.kappa

                self.phi = torch.sum(torch.exp(torch.pow(-self.u + self.kappa, 2) * (-beta)) * alpha, 1, keepdim=True)

                w = torch.squeeze(torch.matmul(self.phi, torch.Tensor(c_vec).to(device)), 1)

                cell2_state = self.rnn_cell2(torch.cat([x[:,t,:], cell1_state[0], w], 1), cell2_state)

                output_list[:, t,:] = cell2_state[0]
            outputs = self.fc_output_syn(output_list.reshape(-1, 512))
        else:
            raise ValueError


        y1, y2, s = torch.unbind(y.view(-1, 3), dim=1)

        e = 1 / (1 + torch.exp(outputs[:, 0]))
        pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat = torch.split(outputs[:, 1:], self.num_gaussians, 1)

        pi = pi_hat.softmax(1)

        sigma1 = torch.exp(sigma1_hat - self.bias)
        sigma2 = torch.exp(sigma2_hat - self.bias)
        rho = torch.tanh(rho_hat)
        gaussian = pi * self._bivariate_gaussian(
            y1.unsqueeze(1).repeat(1, self.num_gaussians), y2.unsqueeze(1).repeat(1, self.num_gaussians),
            mu1, mu2, sigma1, sigma2, rho
        )
        eps = 1e-20
        loss_gaussian = torch.sum(-torch.log(torch.sum(gaussian, 1) + eps))
        loss_bernoulli = torch.sum(-torch.log((e + eps) * s + (1 - e + eps) * (1 - s)))

        self.loss = (loss_gaussian + loss_bernoulli) / (x.shape[0] * self.seq_length)


        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


    def sample(self, batch_size, length, s=None):
        x = torch.zeros([batch_size, 1, 3], dtype=torch.float32).to(device)

        x[:, 0, 2] = 1
        strokes = torch.zeros([batch_size, length, 3], dtype=torch.float32).to(device)
        strokes[:, 0, :] = x[:, 0, :]

        if self.mode == 'prediction':
            states1, states2, states3 = None, None, None
        elif self.mode == 'synthesis':
            cell1_state, cell2_state = None, None
            w = torch.zeros(batch_size, self.c_dimension).to(device)
            kappa_prev = torch.zeros(batch_size, self.K, 1).to(device)
        else:
            raise ValueError

        for i in range(length - 1):
            if self.mode == 'prediction':
                outputs1, states1 = self.lstm1(x, states1)
                outputs2, states2 = self.lstm2(torch.cat([x, outputs1], dim=2), states2)
                outputs3, states3 = self.lstm3(torch.cat([x, outputs2], dim=2), states3)

                outputs = self.fc_output(torch.cat([outputs1, outputs2, outputs3], dim=2).reshape(-1, self.fc_output.in_features))
            elif self.mode == 'synthesis':
                cell1_state = self.rnn_cell1(torch.cat([x[:,0,:], w], 1), cell1_state)
                k_gaussian = self.h2k(cell1_state[0])

                alpha_hat, beta_hat, kappa_hat = torch.split(k_gaussian, self.K, dim=1)

                alpha = torch.exp(alpha_hat).unsqueeze(2)
                beta = torch.exp(beta_hat).unsqueeze(2)

                self.kappa = kappa_prev + torch.exp(kappa_hat).unsqueeze(2)
                kappa_prev = self.kappa

                self.phi = torch.sum(torch.exp(torch.pow(-self.u + self.kappa, 2) * (-beta)) * alpha, 1, keepdim=True)

                w = torch.squeeze(torch.matmul(self.phi, torch.Tensor(s).to(device)), 1)

                cell2_state = self.rnn_cell2(torch.cat([x[:,0,:], cell1_state[0], w], 1), cell2_state)

                output_list = cell2_state[0]

                outputs = self.fc_output_syn(output_list.reshape(-1, 512))
            else:
                raise ValueError
                


            end_of_stroke = 1 / (1 + torch.exp(outputs[:, 0]))
            pi_hat, mu1, mu2, sigma1_hat, sigma2_hat, rho_hat = torch.split(outputs[:, 1:], self.num_gaussians, 1)
            pi = torch.softmax(pi_hat * (1 + self.bias), dim=1)

            sigma1 = torch.exp(sigma1_hat - self.bias)
            sigma2 = torch.exp(sigma2_hat - self.bias)
            rho = torch.tanh(rho_hat)

            x = torch.zeros([batch_size, 1, 3], dtype=torch.float32).to(device)
            for m in range(self.num_gaussians):
                matrix = torch.zeros(sigma1.shape[0], 2, 2).to(device)
                matrix[:, 0, 0] = sigma1[:, m] ** 2
                matrix[:, 0, 1] = matrix[:, 1, 0] = rho[:, m] * sigma1[:, m] * sigma2[:, m]
                matrix[:, 1, 1] = sigma2[:, m] ** 2
                means = torch.zeros(sigma1.shape[0], 2).to(device)
                means[:, 0] = mu1[:, m]
                means[:, 1] = mu2[:, m]
                mn = MultivariateNormal(means, matrix).sample()
                x[:, 0, 0:2] += pi[:, m].unsqueeze(1).repeat(1, 2) * mn

            e = torch.rand(batch_size).to(device)
            slicing = torch.Tensor(list(e<end_of_stroke)) == True
            x[slicing, 0, 2] = 1
            strokes[:, i + 1, :] = x[:, 0, :]

        return strokes
