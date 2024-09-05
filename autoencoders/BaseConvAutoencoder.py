import torch.nn as nn
from torch.nn import functional as F

class BaseConvAutoencoder(nn.Module):
    def __init__(self, windows_size, input_size):
        super(BaseConvAutoencoder, self).__init__()
        self.conv_features = [2, 3, [23, 156, 286], [2, 3, 5], [1, 1, 1], [1, 1, 1], False]

        # Encoder
        self.first_kernel_size = 7
        self.padding_1 = int(self.first_kernel_size / 2) if input_size < 10 else 1

        self.second_kernel_size = 2
        self.padding_2 = int((self.second_kernel_size) / 2) if self.first_kernel_size >= input_size else 1

        self.third_kernel_size = 5
        self.padding_3 = int((self.third_kernel_size /2)) if input_size == 1 else 1


        #####SMAP
        self.conv1 = nn.Conv1d(in_channels=windows_size, out_channels=75, kernel_size=3, padding=self.padding_1)
        self.conv1_bn = nn.BatchNorm1d(num_features=75)
        self.conv2 = nn.Conv1d(in_channels=75, out_channels=70, kernel_size=2, padding=1)
        self.conv2_bn = nn.BatchNorm1d(num_features=70)
        self.conv3 = nn.Conv1d(in_channels=70, out_channels=273, kernel_size=5, padding=self.padding_3)
        self.conv3_bn = nn.BatchNorm1d(num_features=273)
        self.conv4 = nn.Conv1d(in_channels=273, out_channels=437, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm1d(num_features=437)
        self.conv5 = nn.Conv1d(in_channels=437, out_channels=694, kernel_size=2, padding=1)
        self.conv5_bn = nn.BatchNorm1d(num_features=694)
        self.conv6 = nn.Conv1d(in_channels=694, out_channels=1041, kernel_size=2, padding=1)
        self.conv6_bn = nn.BatchNorm1d(num_features=1041)

        self.t_conv1 = nn.ConvTranspose1d(in_channels=1041, out_channels=694, kernel_size=2,
                                          padding=self.padding_1)
        self.t_conv1_bn = nn.BatchNorm1d(num_features=694)
        self.t_conv2 = nn.ConvTranspose1d(in_channels=694, out_channels=437, kernel_size=2, padding=1)
        self.t_conv2_bn = nn.BatchNorm1d(num_features=437)
        self.t_conv3 = nn.ConvTranspose1d(in_channels=437, out_channels=273, kernel_size=3, padding=1)
        self.t_conv3_bn = nn.BatchNorm1d(num_features=273)
        self.t_conv4 = nn.ConvTranspose1d(in_channels=273, out_channels=70, kernel_size=5, padding=self.padding_3)
        self.t_conv4_bn = nn.BatchNorm1d(num_features=70)
        self.t_conv5 = nn.ConvTranspose1d(in_channels=70, out_channels=75, kernel_size=2, padding=1)
        self.t_conv5_bn = nn.BatchNorm1d(num_features=75)
        self.t_conv6 = nn.ConvTranspose1d(in_channels=75, out_channels=windows_size, kernel_size=3, padding=1)
        self.t_conv6_bn = nn.BatchNorm1d(num_features=windows_size)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv1_bn(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv2_bn(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv3_bn(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv4_bn(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv5_bn(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv6_bn(x))

        x = F.leaky_relu(self.t_conv1(x))
        x = F.leaky_relu(self.t_conv1_bn(x))
        x = F.leaky_relu(self.t_conv2(x))
        x = F.leaky_relu(self.t_conv2_bn(x))
        x = F.leaky_relu(self.t_conv3(x))
        x = F.leaky_relu(self.t_conv3_bn(x))
        x = F.leaky_relu(self.t_conv4(x))
        x = F.leaky_relu(self.t_conv4_bn(x))
        x = F.leaky_relu(self.t_conv5(x))
        x = F.leaky_relu(self.t_conv5_bn(x))
        x = self.t_conv6(x)
        x = self.t_conv6_bn(x)

        return x

