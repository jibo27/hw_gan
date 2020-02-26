import torch
import torch.nn.functional as F
import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Discriminator(torch.nn.Module):
    def __init__(self, learning_rate, weight_decay):
        super(Discriminator, self).__init__()

        self.conv1 = torch.nn.Sequential(
                # layer1
                # input: (batch_size, 7, height=128, width=w)
                torch.nn.Conv2d(
                    in_channels = 7,
                    out_channels = 32,
                    kernel_size = 3,
                    padding = 1,
                    stride = 1),
                torch.nn.BatchNorm2d(32),
                torch.nn.LeakyReLU(),
                # output: (batch_size, 32, 128, w)
                torch.nn.AvgPool2d(kernel_size = 2), 
                # output: (batch_size, 32, 64, w / 2)


                # layer2
                # input: (batch_size, 32, 64, w / 2)
                torch.nn.Conv2d(
                    in_channels = 32,
                    out_channels = 64,
                    kernel_size = 3,
                    padding = 1,
                    stride = 1),
                torch.nn.BatchNorm2d(64),
                torch.nn.LeakyReLU(),
                # output: (batch_size, 64, 64, w / 2)
                torch.nn.AvgPool2d(kernel_size = 2), 


                # layer3
                # input: (batch_size, 64, 32, w / 4)
                torch.nn.Conv2d(
                    in_channels = 64,
                    out_channels = 128,
                    kernel_size = 3,
                    padding = 1,
                    stride = 1),
                torch.nn.BatchNorm2d(128),
                torch.nn.LeakyReLU(),
                # output: (batch_size, 128, 32, w / 4)

                # layer4
                # input: (batch_size, 128, 32, w / 4)
                torch.nn.Conv2d(
                    in_channels = 128,
                    out_channels = 256,
                    kernel_size = 3,
                    padding = 1,
                    stride = (2, 1)),
                torch.nn.BatchNorm2d(256),
                torch.nn.LeakyReLU(),
                # output: (batch_size, 256, 16, w / 4)
                torch.nn.AvgPool2d(kernel_size = 2), 
                # output: (batch_size, 256, 8, w / 8)


                # layer5
                # input: (batch_size, 256, 8, w / 8)
                torch.nn.Conv2d(
                    in_channels = 256,
                    out_channels = 128,
                    kernel_size = 3,
                    padding = 1,
                    stride = (2, 1)),
                torch.nn.BatchNorm2d(128),
                torch.nn.LeakyReLU(),
                # output: (batch_size, 128, 4, w / 8)

                # layer6
                # input: (batch_size, 128, 4, w / 8)
                torch.nn.Conv2d(
                    in_channels = 128,
                    out_channels = 256,
                    kernel_size = 3,
                    padding = 1,
                    stride = (2, 1)),
                torch.nn.BatchNorm2d(256),
                torch.nn.LeakyReLU(),
                # output: (batch_size, 256, 2, w / 8)
                torch.nn.AvgPool2d(kernel_size = 2),
                # output: (batch_size, 256, 1, w / 16)
                )

        self.lstm1 = torch.nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        self.fc1 = torch.nn.Linear(256, 100)
        self.leaky_relu1 = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(100, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def predict(self, net, widths):
        net = self.conv1(net) # (batch_size, 7, 1, width)
        net = net.view(net.size(0), -1, net.size(1)) # reshape => (batch_size, width, 7)
        net_out, (h_n, h_c) = self.lstm1(net, None)
        net = torch.stack([output[widths[index] // 16 - 1] for index, output in enumerate(net_out)])
        net = self.fc1(net)
        net = self.leaky_relu1(net)
        net = self.fc2(net)
        net = torch.sigmoid(net)
        return net
