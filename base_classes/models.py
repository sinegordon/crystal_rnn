import torch
import torch.nn as nn


class RNNAutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, self.hidden_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x)
        x = self.decoder(h)
        return x, h


class RNNNet(nn.Module):
    def __init__(self, in_features, hidden_size, num_layers, autoencoder=None, type="RNN"):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_features = in_features
        self.out_features = in_features
        self.num_layers = num_layers
        self.rnn_type = type.upper()
        self.encoder = None
        self.decoder = None

        if autoencoder is not None:
            self.encoder = autoencoder.encoder
            self.decoder = autoencoder.decoder

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                in_features,
                self.hidden_size,
                batch_first=True,
                num_layers=num_layers,
                bidirectional=True,
            )
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                in_features,
                self.hidden_size,
                batch_first=True,
                num_layers=num_layers,
                bidirectional=True,
            )
        else:
            self.rnn = nn.RNN(
                in_features,
                self.hidden_size,
                batch_first=True,
                num_layers=num_layers,
                bidirectional=True,
            )

        self.out = nn.Linear(self.hidden_size * 2, self.out_features)

    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)

        x, h = self.rnn(x)
        if isinstance(h, tuple):
            h = h[0]

        hh = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        z = self.out(hh)

        if self.decoder is not None:
            z = self.decoder(z)
        return z
