import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class InputData():
    def __init__(self, X, Y):
        self.x = torch.FloatTensor(X)
        self.y = torch.FloatTensor(Y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


class LSTM_Model(nn.Module):
    def __init__(self, input_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=16, num_layers=2)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def read_data():
    df = pd.read_csv(filepath_or_buffer="LSTM/airline-passengers.csv", header=1, names=["Month","Passengers"])
    data = df[["Passengers"]].values
    scaler = StandardScaler()
    s_data = scaler.fit_transform(data)
    y = []
    x = []
    for i in range(0, len(s_data), 1):
        try:
            x_out = s_data[i:(i+10)].tolist()
            y_out = s_data[(i+10)].tolist()
            x.append(x_out)
            y.append(y_out)
        except IndexError:
            pass
    return x, y



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = read_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.2)
    training_data = InputData(X=x_train, Y=y_train)
    testing_data = InputData(X=x_test, Y=y_test)

    training_loader = DataLoader(dataset=training_data, batch_size=10)
    testing_loader = DataLoader(dataset=testing_data, batch_size=1)

    model = LSTM_Model().to(device)
    epoches = 10000
    learning_rate = 0.0001
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    total_loss = []
    diff_loss_count = 0

    for i in range(epoches):
        batch_loss = 0
        for x, y in training_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(x)
            loss = loss_function(y_pred, y)

            loss.backward()
            optimizer.step()

            batch_loss += loss
        avg_loss = batch_loss/len(training_loader)
        total_loss.append(avg_loss.cpu().detach().numpy())

        if i > 2 and total_loss[i-1] - avg_loss.cpu().detach().numpy() < 0.00001:
            diff_loss_count += 1
        else:
            diff_loss_count = 0

        if diff_loss_count >= 5:
            break

        print(f"epoch {i} avg_loss {avg_loss} lr={learning_rate} early_stop_count={diff_loss_count}")


    plt.plot(total_loss)
    plt.show()

    with torch.no_grad():
        pred = []
        for xt, yt in testing_loader:
            xt = xt.to(device)
            yt = yt.to(device)
            y_eval = model.forward(xt)
            eval_loss = loss_function(y_eval, yt)
            print(f"test_loss {eval_loss} prediction {y_eval[0]}")
            pred.append(y_eval[0].cpu().detach().numpy())

    plt.plot(pred)
    plt.show()

    torch.save(obj=model, f="LSTM/lstm_time_series.pt")



