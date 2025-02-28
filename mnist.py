import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Loading data from a Folder
# training_data = datasets.ImageFolder(root="MNIST/train", transform=tramsform)


class CNN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=1, kernel_size=3, stride=1, out_channels=32)
        self.c11 = nn.Conv2d(in_channels=32, kernel_size=3, stride=1, out_channels=64)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.c2 = nn.Conv2d(in_channels=64, kernel_size=3, stride=1, out_channels=128)
        self.c22 = nn.Conv2d(in_channels=128, kernel_size=3, stride=1, out_channels=256)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=256*4*4, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=128, out_features=10)



    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c11(x))
        x = self.p1(x)
        x = F.relu(self.c2(x))
        x = F.relu(self.c22(x))
        x = self.p2(x)
        x = x.view(-1, 256*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dp1(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tramsform = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])
    training_data = datasets.MNIST(root=".", train=True, download=True, transform=tramsform)
    testing_data = datasets.MNIST(root=".", train=False, download=True, transform=tramsform)

    training_loader = DataLoader(dataset=training_data, batch_size=128, shuffle=True)
    testing_loader = DataLoader(dataset=testing_data, batch_size=1, shuffle=False)

    epoches = 200
    model = CNN_Net().to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    total_loss = []
    early_stopping = 0

    for epoch in range(epoches):
        batch_loss = 0
        diff_avg_loss = 0
        for i, (x, y) in enumerate(training_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(x)
            loss = criterion(y_pred, y)
            batch_loss = batch_loss + loss

            loss.backward()
            optimizer.step()
            # print(f"Mini batch ID {i} ---- Batch loss {loss}")

        batch_length = len(training_loader)
        avg_batch_loss = batch_loss/batch_length

        if epoch > 2:
            diff_avg_loss = total_loss[epoch-1] - avg_batch_loss.cpu().detach().numpy()
            print(diff_avg_loss)

        if diff_avg_loss < 0.001:
            early_stopping += 1

        if early_stopping >= 5:
            break

        total_loss.append(avg_batch_loss.cpu().detach().numpy())

        print(f"Epoch {epoch} avg_loss {avg_batch_loss} batch length {batch_length} diff_avg_loss {diff_avg_loss}")

    print(total_loss)

    plt.plot(total_loss)
    plt.show()

    with torch.no_grad():
        for i, (x, y) in enumerate(testing_loader):
            x, y = x.to(device), y.to(device)
            y_eval = model.forward(x)
            test_loss = criterion(y_eval, y)
            print(f"Prediction {y_eval.argmax()} -- actual {y}")

    torch.save(model, f="MNIST/mnist_model.pt")





