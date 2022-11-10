import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transfroms

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1)
if device == 'cuda':
    torch.cuda.manual_seed_all(1)
print(device + " is available")

learning_rate = 1e-3
batch_size = 256
num_classes = 10
nb_epochs = 100


class model_1(nn.Module):
    def __init__(self):
        super(model_1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (5, 5), stride=(1, 1))
        self.mp1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, (3, 3), stride=(1, 1))
        self.mp2 = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.fc1 = nn.Linear(800, 400)
        self.fc2 = nn.Linear(400, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.mp2(x)
        x = self.relu(x)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x


def plot_one(nb_epochs, costs, accuracies):
    costs = np.array(costs)
    accuracies = np.array(accuracies)
    
    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    
    ax1 = fig.add_subplot(121)
    ax1.set_ylabel('cost')
    ax1.set_xlabel('epochs')
    ax1.set_title('Costs for epochs')
    ax1.plot(np.arange(1, nb_epochs + 1, 1), costs)

    ax2 = fig.add_subplot(122)
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epochs')
    ax2.set_title('Accuracies for epochs')
    ax2.plot(np.arange(1, nb_epochs + 1, 1), accuracies)    

    plt.show()


def load_mnist():
    train_set = torchvision.datasets.MNIST(
        root = '/datasets/MNIST/raw',
        train = True,
        download = True,
        transform = transfroms.Compose([
            transfroms.ToTensor()
        ])
    )
    test_set = torchvision.datasets.MNIST(
        root = '/datasets/MNIST/raw',
        train = False,
        download = True,
        transform = transfroms.Compose([
            transfroms.ToTensor()
        ])
    )
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    return train_loader, test_loader


def main():
    train_loader, test_loader = load_mnist()

    ##### MODEL 1 #####
    model1 = model_1().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model1.parameters(), lr=learning_rate)

    model1_cost_list = []
    model1_accuracy_list = []
    for epoch in tqdm(range(nb_epochs)):
        epoch_cost = 0
        correct = 0
        total = 0
        for data, label in train_loader:
            optimizer.zero_grad()   # Set gradient to "0"

            label = label.to(device)
            data = data.to(device)  # [256, 1, 28, 28]
            # data = data.view(data.shape[0], -1)
            hypothesis = model1(data)

            hypo_label = torch.max(hypothesis.data, 1)[1]
            correct += (hypo_label == label).sum().item()
            total += len(label)

            cost = criterion(hypothesis, label)
            cost.backward()
            optimizer.step()
            epoch_cost += cost.item()

        epoch_cost /= len(train_loader)
        accuracy = correct / total * 100

        model1_cost_list.append(epoch_cost)
        model1_accuracy_list.append(accuracy)

        if epoch % 10 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch, nb_epochs, epoch_cost, accuracy))

    with torch.no_grad():
        correct = 0
        total = 0

        for data, label in tqdm(test_loader):
            label = label.to(device)
            data = data.to(device)
            preds = model1(data)

            preds_label = torch.max(preds.data, 1)[1]
            correct += (preds_label == label).sum().item()
            total += len(label)

        accuracy = correct / total * 100
        print('Test Accuracy: ', accuracy, '%')

    plot_one(nb_epochs, model1_cost_list, model1_accuracy_list)

if __name__ == '__main__':
    main()