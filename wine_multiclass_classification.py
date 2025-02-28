import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


def load_data(infile="wine/wine.csv"):
    df = pd.read_csv(filepath_or_buffer=infile, sep=";")
    df["quality"] = df["quality"].map({3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6})
    x = df.drop("quality", axis=1).values
    y = df["quality"].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    return x_train, x_test, y_train, y_test


class WineDataLoader():
    def __init__(self, X, Y):
        self.x = torch.FloatTensor(X)
        self.y  = torch.LongTensor(Y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


class Model(nn.Module):
    def __init__(self, f_input=11, f_output=7):
        super().__init__()
        self.inputlayer = nn.Linear(in_features=f_input, out_features=128)
        self.f1 = nn.Linear(in_features=128, out_features=64)
        self.f2 = nn.Linear(in_features=64, out_features=32)
        self.outputlayer = nn.Linear(in_features=32, out_features=f_output)

    def forward(self, x):
        x = F.relu(self.inputlayer(x))
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.log_softmax(self.outputlayer(x), dim=1)
        return x



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, x_test, y_train, y_test = load_data()

    training_data = WineDataLoader(X=x_train, Y=y_train)
    trainingdata_loader = DataLoader(dataset=training_data, batch_size=16, shuffle=True)

    model = Model().to(device)
    model = model

    epoches = 1500
    optimizer = optimizer.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    total_loss = []
    # early_stopping = 0

    for epoch in range(epoches):
        batch_loss = 0
        training_counter = 0
        for x, y in trainingdata_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            y_pred = model.forward(x)
            loss = criterion(y_pred, y)
            batch_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = batch_loss/len(trainingdata_loader)
        total_loss.append(avg_loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epoches} avg_loss {avg_loss}")


    plt.plot(total_loss)
    plt.show()

    with torch.no_grad():
        test_data = WineDataLoader(X=x_test, Y=y_test)
        test_loader = DataLoader(dataset=test_data, batch_size=1)

        test_counter = 0
        for x_, y_ in test_loader:
            y_eval = model.forward(x_)
            test_loss = criterion(y_eval, y_)
            print(f"DataSet {test_counter} test_loss {test_loss.item()} predicted {y_eval.argmax()} truth {y_.item()}")
            test_counter += 1


    torch.save(model, f="wine/wide_multiclass_classifier.pt")

