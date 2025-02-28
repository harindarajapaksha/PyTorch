import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



class Mydata():
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(Y)
        self.len = self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len


class Model(nn.Module):
    def __init__(self, f_input=4, f_output=3):
        super().__init__()
        self.f_input = nn.Linear(in_features=f_input, out_features=128)
        self.f1 = nn.Linear(in_features=128, out_features=64)
        self.f2 = nn.Linear(in_features=64, out_features=32)
        self.f_out = nn.Linear(in_features=32, out_features=f_output)

    def forward(self, x):
        x = F.relu(self.f_input(x))
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.log_softmax(self.f_out(x), dim=1)

        return x


def iris_dataset():
    df = load_iris()
    x = df.data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    y = df.target
    return x, y


if __name__ == '__main__':

    x, y = iris_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    data_train = Mydata(X=x_train, Y=y_train)
    training_loader = DataLoader(dataset=data_train, batch_size=8)

    model = Model()
    model.train()
    epoches = 1000
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    running_loss = []

    for ephoch in range(epoches):
        ephoch_loss = 0.0
        epoch_counter = 0
        for x, y in training_loader:
            optimizer.zero_grad()
            y_pred = model.forward(x)
            loss = criterion(y_pred, y)
            ephoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            epoch_counter += 1

        if ephoch % 100 == 0:
            print(f"Epoch {ephoch}/{epoches} avg_training_loss {ephoch_loss/len(training_loader)} ")
        running_loss.append(ephoch_loss/len(training_loader))

    plt.plot(running_loss)
    plt.show()

    x_test = torch.FloatTensor(x_test)

    data_test = Mydata(X=x_test, Y=y_test)
    testing_loader = DataLoader(dataset=data_test, batch_size=1)

    with torch.no_grad():
        test_counter = 0
        y_eval_list = []
        for x, y in testing_loader:
            y_eval = model.forward(x)
            y_eval_list.append(y_eval)
            loss_eval = criterion(y_eval, y)
            print(f"Data_set_index {test_counter} test_loss {loss_eval} -> y_pred {y_eval.argmax()} y_true {y.item()}")
            test_counter += 1

    torch.save(model, f="iris/IrisModel.pt")

