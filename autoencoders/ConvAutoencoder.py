import torch.nn as nn
from torch.nn import functional as F
import math


class ConvAutoencoder(nn.Module):
    def __init__(self, input_size, windows_size, nr_of_conv_layers, channels, kernels, strides, paddings, number_of_fc_layer,
                 fc_layers_prec_output_size, loss_function, optimizer, learning_rate, skip_connections=0):
        super(ConvAutoencoder, self).__init__()

        self.input_size = input_size
        self.window = windows_size
        self.nr_of_conv_layers = nr_of_conv_layers
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.paddings = paddings
        self.conv_features = [windows_size, nr_of_conv_layers, channels, kernels, strides, paddings, skip_connections]
        self.number_of_fc_layer = number_of_fc_layer
        self.fc_layers_prec_output_size = fc_layers_prec_output_size
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.shape_of_fc_input = 0
        self.shape_of_fc_decoder_out = []

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(nr_of_conv_layers):
            if i == 0:
                paddings[i] = 1 if self.input_size >= self.kernels[i] else  int(kernels[i] / 2)
                self.shape_of_fc_input = conv_output_shape(input_size, kernels[i], strides[i], paddings[i])
                self.encoder.append(
                    nn.Conv1d(in_channels=windows_size, out_channels=channels[i], kernel_size=kernels[i],
                              padding=paddings[i]))
            else:
                paddings[i] = 1 if self.shape_of_fc_input >= kernels[i] else  int(kernels[i] / 2)
                self.encoder.append(
                    nn.Conv1d(channels[i - 1], out_channels=channels[i], kernel_size=kernels[i], padding=paddings[i]))
                self.shape_of_fc_input = conv_output_shape(self.shape_of_fc_input, kernels[i], strides[i], paddings[i])

            self.encoder.append(nn.BatchNorm1d(num_features=channels[i]))
            self.encoder.append(nn.ReLU())

        for i in range(self.number_of_fc_layer):
            # nr_of_fc_output_channel = 0
            nr_of_fc_input_channel = self.shape_of_fc_input * channels[
                nr_of_conv_layers - 1] if i == 0 else nr_of_fc_output_channel
            nr_of_fc_output_channel = int(self.fc_layers_prec_output_size[i] * nr_of_fc_input_channel)
            self.encoder.append(nn.Linear(nr_of_fc_input_channel, nr_of_fc_output_channel))

        for i in range(self.number_of_fc_layer):
            out_features = self.encoder[len(self.encoder) - 1 - i].in_features
            in_features = self.encoder[len(self.encoder) - 1 - i].out_features
            self.decoder.append(nn.Linear(in_features, out_features))

        for i in range(nr_of_conv_layers):
            # paddings[-i - 1] = 1 if self.shape_of_fc_input >= kernels[-i - 1] else \
            #     (int(kernels[-i - 1] / 2) if int(kernels[-i - 1] / 2) > self.shape_of_fc_input else int(kernels[-i - 1] / 2 + 1))
            paddings[-i - 1] = paddings[nr_of_conv_layers - 1 - i]
            if i == nr_of_conv_layers - 1:
                self.decoder.append(nn.ConvTranspose1d(in_channels=channels[-i - 1], out_channels=windows_size,
                                                       kernel_size=kernels[-i - 1], padding=paddings[-i - 1]))
            else:
                self.decoder.append(nn.ConvTranspose1d(in_channels=channels[-i - 1], out_channels=channels[-i - 2],
                                                       kernel_size=kernels[-i - 1], padding=paddings[-i - 1]))
                self.decoder.append(nn.BatchNorm1d(num_features=channels[-i - 2]))
            self.decoder.append(nn.ReLU())

    def forward(self, x):

        for i in range(len(self.encoder)):
            if isinstance(self.encoder[i], nn.modules.Linear):
                self.shape_of_fc_decoder_out.append(x.shape)
                x = x.view(-1, x.shape[1] * x.shape[2])
                # if self.encoder[i]

            x = self.encoder[i](x)

        index = 0
        for i in range(len(self.decoder)):
            if isinstance(self.decoder[i], nn.modules.ConvTranspose1d) and isinstance(self.decoder[i - 1],
                                                                                      nn.modules.Linear):
                x = x.view(self.shape_of_fc_decoder_out[index][0], self.shape_of_fc_decoder_out[index][1],
                           self.shape_of_fc_decoder_out[index][2])
                index = index + 1
            x = self.decoder[i](x)

        return x


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    h = (h_w + (2 * pad) - (dilation * (kernel_size - 1)) - 1) // stride + 1

    return h


def convtransp_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of transposed convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """

    if type(h_w) is not tuple:
        h_w = (h_w, h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0] + pad[0]
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1] + pad[1]

    return h, w
