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


class FrameLayerRNNNet(nn.Module):
    """Flat recurrent model with one recurrent cell per history frame.

    ``nn.RNN(num_layers=N)`` means a stack of N recurrent layers, and every
    layer still sees the whole input sequence.  This model is different: frame
    ``t`` is processed by its own cell ``cell_t``, then the hidden state is
    passed to the next frame-specific cell.  For bidirectional mode the same
    idea is mirrored from the last frame to the first.
    """

    def __init__(
        self,
        in_features,
        hidden_size,
        num_layers,
        autoencoder=None,
        type="RNN",
        bidirectional=True,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.in_features = int(in_features)
        self.out_features = self.in_features
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.rnn_type = type.upper()
        self.bidirectional = bool(bidirectional)
        self.encoder = None
        self.decoder = None

        if autoencoder is not None:
            self.encoder = autoencoder.encoder
            self.decoder = autoencoder.decoder

        if self.rnn_type == "GRU":
            cell_cls = nn.GRUCell
        elif self.rnn_type == "LSTM":
            cell_cls = nn.LSTMCell
        elif self.rnn_type == "RNN":
            cell_cls = nn.RNNCell
        else:
            raise ValueError("type must be 'RNN', 'GRU', or 'LSTM'")

        self.forward_cells = nn.ModuleList(
            cell_cls(self.in_features, self.hidden_size) for _ in range(self.num_layers)
        )
        if self.bidirectional:
            self.backward_cells = nn.ModuleList(
                cell_cls(self.in_features, self.hidden_size) for _ in range(self.num_layers)
            )
            readout_width = self.hidden_size * 2
        else:
            self.backward_cells = None
            readout_width = self.hidden_size
        self.out = nn.Linear(readout_width, self.out_features)

    def _initial_state(self, batch_size, device, dtype):
        hidden = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        if self.rnn_type == "LSTM":
            cell = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
            return hidden, cell
        return hidden

    def _run_cells(self, x, cells, reverse=False):
        state = self._initial_state(x.shape[0], x.device, x.dtype)
        indices = range(self.num_layers - 1, -1, -1) if reverse else range(self.num_layers)
        for cell_index in indices:
            state = cells[cell_index](x[:, cell_index, :], state)
        if self.rnn_type == "LSTM":
            return state[0]
        return state

    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        if x.ndim != 3:
            raise ValueError("FrameLayerRNNNet expects input shape (batch, sequence, features)")
        if x.shape[1] != self.num_layers:
            raise ValueError("FrameLayerRNNNet sequence length must equal num_layers")

        forward_hidden = self._run_cells(x, self.forward_cells, reverse=False)
        if self.bidirectional:
            backward_hidden = self._run_cells(x, self.backward_cells, reverse=True)
            readout = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            readout = forward_hidden
        z = self.out(readout)

        if self.decoder is not None:
            z = self.decoder(z)
        return z
