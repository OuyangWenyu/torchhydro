import torch
from torch import nn


class TimeDistributed(nn.Module):
    def __init__(self, layer):
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        outputs = []
        for t in range(x.size(1)):
            xt = x[:, t, :]
            output = self.layer(xt)
            outputs.append(output.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs


class SppLayer(nn.Module):
    def __init__(self, out_pool_size):
        """
        out_pool_size: a int vector of expected output size of max pooling layer
        """
        super(SppLayer, self).__init__()
        self.out_pool_size = out_pool_size
        self.pools = []
        for i in range(len(out_pool_size)):
            pool_i = nn.AdaptiveMaxPool2d(out_pool_size[i])
            self.pools.append(pool_i)

    def forward(self, previous_conv):
        """
        Parameters
        ----------
        previous_conv
            a tensor vector of previous convolution layer

        Returns
        -------
        torch.Tensor
            a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        """

        num_sample = previous_conv.size(0)
        channel_size = previous_conv.size(1)
        out_pool_size = self.out_pool_size
        for i in range(len(out_pool_size)):
            maxpool = self.pools[i]
            x = maxpool(previous_conv)
            if i == 0:
                spp = x.view(num_sample, channel_size, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, channel_size, -1)), -1)
        return spp


class SPP_LSTM_Model(nn.Module):
    def __init__(self, seq_length, n_output, n_hidden_states):
        super(SPP_LSTM_Model, self).__init__()

        self.conv1 = TimeDistributed(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=1,
                padding="same",
            )
        )

        self.conv2 = TimeDistributed(
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                padding="same",
                bias=True,
            )
        )

        self.conv3 = TimeDistributed(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding="same",
                bias=True,
            )
        )

        self.maxpool1 = TimeDistributed(nn.MaxPool2d(kernel_size=2, stride=(2, 2)))

        self.conv4 = TimeDistributed(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding="same",
                bias=True,
            )
        )

        self.conv5 = TimeDistributed(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding="same",
                bias=True,
            )
        )

        self.maxpool2 = TimeDistributed(SppLayer([4, 2, 1]))

        self.dropout = nn.Dropout(p=0.1)

        self.lstm = nn.LSTM(
            input_size=21 * 32 * seq_length, hidden_size=n_hidden_states
        )

        self.dense = nn.Linear(in_features=n_hidden_states, out_features=n_output)

    def forward(self, x):
        # print(x.shape)
        x = torch.squeeze(x, dim=1)
        # print(x.shape)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.dropout(x)
        x = x.view(len(x), 1, -1)
        # print(x.shape)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dense(x)
        # print(x.shape)
        return x.unsqueeze(2).transpose(0, 1)
