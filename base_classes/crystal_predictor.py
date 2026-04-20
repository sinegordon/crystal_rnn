import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from .datasets import RNNCustomDataset
from .models import RNNNet


class CrystalRNNNet:
    def __init__(self, in_features, hidden_size, num_layers, type="RNN"):
        super().__init__()
        self.in_features = in_features
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_type = type.upper()
        self.model = RNNNet(self.in_features, self.hidden_size, num_layers, type=self.rnn_type)
        self.lr = 0.001
        self.epochs = 50
        self.batch_size = 200
        self.train_count = 200

    def reset(self):
        self.model = RNNNet(self.in_features, self.hidden_size, self.num_layers, type=self.rnn_type)

    def train(self, X_coords, y_coords, data_len=0.5):
        self.train_count = int(data_len * X_coords.shape[0])
        if self.train_count <= 0:
            raise ValueError("train_count must be positive")
        if self.train_count >= X_coords.shape[0]:
            ind = 0
        else:
            ind = np.random.randint(low=0, high=X_coords.shape[0] - self.train_count)

        X_train = X_coords[ind : ind + self.train_count]
        y_train = y_coords[ind : ind + self.train_count]
        train_dataset = RNNCustomDataset(X_train, y_train)
        train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        losses = []

        for _ in tqdm.trange(self.epochs):
            loss_mean = 0
            lm_count = 0

            for x_train, y_train in train_data:
                predict = self.model(x_train)
                loss = loss_func(predict, y_train)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lm_count += 1
                loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean

            losses.append(loss_mean)

        return losses

    def run(self, count_steps, init_features):
        self.model.eval()
        mas = []
        x = torch.as_tensor(init_features, dtype=torch.float32)

        with torch.no_grad():
            for _ in range(count_steps):
                y = self.model(x).squeeze(0)
                mas.append(y.detach().cpu().numpy())
                x[0] = torch.vstack([x[0, 1:], y])

        mas = np.array(mas)
        return mas


class CrystalRNNNetBagging:
    def __init__(self, models, in_features):
        self.models = [model.model for model in models]
        self.in_features = in_features

    def run(self, count_steps, init_features, separate=False):
        for model in self.models:
            model.eval()

        mas = []
        if separate:
            for model in self.models:
                x = torch.as_tensor(init_features, dtype=torch.float32)
                mas1 = []
                with torch.no_grad():
                    for _ in range(count_steps):
                        y = model(x).squeeze(0)
                        mas1.append(y.detach().cpu().numpy())
                        x[0] = torch.vstack([x[0, 1:], y])
                mas.append(mas1)
            mas = np.array(mas)
            mas = mas.mean(axis=0)
        else:
            x = torch.as_tensor(init_features, dtype=torch.float32)
            with torch.no_grad():
                for _ in range(count_steps):
                    y = torch.zeros(len(self.models), self.in_features, dtype=x.dtype)
                    for i, model in enumerate(self.models):
                        y[i, :] = model(x).squeeze(0)
                    z = torch.mean(y, dim=0)
                    mas.append(z.detach().cpu().numpy())
                    x[0] = torch.vstack([x[0, 1:], z])

        mas = np.array(mas)
        return mas
