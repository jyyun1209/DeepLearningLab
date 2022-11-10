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
        self.linear = nn.Linear(28*28, 10)

    def forward(self, x):
        x = self.linear(x)

        return x

class model_2(nn.Module):
    def __init__(self):
        super(model_2, self).__init__()
        self.hidden1 = nn.Linear(28*28, 14*14)
        self.hidden2 = nn.Linear(14*14, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.hidden2(x)

        return x

class model_3(nn.Module):
    def __init__(self):
        super(model_3, self).__init__()
        self.hidden1 = nn.Linear(28*28, 14*14)
        self.hidden2 = nn.Linear(14*14, 7*7)
        self.hidden3 = nn.Linear(7*7, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.hidden2(x)
        x = self.sigmoid(x)
        x = self.hidden3(x)

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


def plot_all(nb_epochs, model1_cost_list, model1_accuracy_list, model2_cost_list, model2_accuracy_list, model3_cost_list, model3_accuracy_list):
    model1_cost_list = np.array(model1_cost_list)
    model2_cost_list = np.array(model2_cost_list)
    model3_cost_list = np.array(model3_cost_list)
    model1_accuracy_list = np.array(model1_accuracy_list)
    model2_accuracy_list = np.array(model2_accuracy_list)
    model3_accuracy_list = np.array(model3_accuracy_list)
    
    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    
    ax1 = fig.add_subplot(121)
    ax1.set_ylabel('cost')
    ax1.set_xlabel('epochs')
    ax1.set_title('Costs for epochs')
    ax1.plot(np.arange(1, nb_epochs + 1, 1), model1_cost_list, 'r', label='0-hidden')
    ax1.plot(np.arange(1, nb_epochs + 1, 1), model2_cost_list, 'g', label='1-hidden')
    ax1.plot(np.arange(1, nb_epochs + 1, 1), model3_cost_list, 'b', label='2-hidden')
    ax1.legend()
    
    ax2 = fig.add_subplot(122)
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epochs')
    ax2.set_title('Accuracies for epochs')
    ax2.plot(np.arange(1, nb_epochs + 1, 1), model1_accuracy_list, 'r', label='0-hidden')
    ax2.plot(np.arange(1, nb_epochs + 1, 1), model2_accuracy_list, 'g', label='1-hidden')
    ax2.plot(np.arange(1, nb_epochs + 1, 1), model3_accuracy_list, 'b', label='2-hidden')
    ax2.legend()

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
    optimizer = optim.SGD(model1.parameters(), lr=learning_rate)

    model1_cost_list = []
    model1_accuracy_list = []
    for epoch in tqdm(range(nb_epochs)):
        epoch_cost = 0
        correct = 0
        total = 0
        for data, label in train_loader:
            optimizer.zero_grad()   # Set gradient to "0"

            label = label.to(device)
            data = data.view(data.shape[0], -1).to(device)
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

        # if epoch % 10 == 0:
        #     print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch, nb_epochs, epoch_cost, accuracy))

    with torch.no_grad():
        correct = 0
        total = 0

        for data, label in tqdm(test_loader):
            label = label.to(device)
            data = data.view(data.shape[0], -1).to(device)
            preds = model1(data)

            preds_label = torch.max(preds.data, 1)[1]
            correct += (preds_label == label).sum().item()
            total += len(label)

        accuracy = correct / total * 100
        print('Test Accuracy: ', accuracy, '%')


    ##### MODEL 2 #####
    model2 = model_2().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model2.parameters(), lr=learning_rate)

    model2_cost_list = []
    model2_accuracy_list = []
    for epoch in tqdm(range(nb_epochs)):
        epoch_cost = 0
        correct = 0
        total = 0
        for data, label in train_loader:
            optimizer.zero_grad()   # Set gradient to "0"

            label = label.to(device)
            data = data.view(data.shape[0], -1).to(device)
            # data = data.view(data.shape[0], -1)
            hypothesis = model2(data)

            hypo_label = torch.max(hypothesis.data, 1)[1]
            correct += (hypo_label == label).sum().item()
            total += len(label)

            cost = criterion(hypothesis, label)
            cost.backward()
            optimizer.step()
            epoch_cost += cost.item()

        epoch_cost /= len(train_loader)
        accuracy = correct / total * 100

        model2_cost_list.append(epoch_cost)
        model2_accuracy_list.append(accuracy)

        # if epoch % 10 == 0:
        #     print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch, nb_epochs, epoch_cost, accuracy))

    with torch.no_grad():
        correct = 0
        total = 0

        for data, label in tqdm(test_loader):
            label = label.to(device)
            data = data.view(data.shape[0], -1).to(device)
            preds = model2(data)

            preds_label = torch.max(preds.data, 1)[1]
            correct += (preds_label == label).sum().item()
            total += len(label)

        accuracy = correct / total * 100
        print('Test Accuracy: ', accuracy, '%')


    ##### MODEL 3 #####
    model3 = model_3().to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model3.parameters(), lr=learning_rate)

    model3_cost_list = []
    model3_accuracy_list = []
    for epoch in tqdm(range(nb_epochs)):
        epoch_cost = 0
        correct = 0
        total = 0
        for data, label in train_loader:
            optimizer.zero_grad()   # Set gradient to "0"

            label = label.to(device)
            data = data.view(data.shape[0], -1).to(device)
            # data = data.view(data.shape[0], -1)
            hypothesis = model3(data)

            hypo_label = torch.max(hypothesis.data, 1)[1]
            correct += (hypo_label == label).sum().item()
            total += len(label)

            cost = criterion(hypothesis, label)
            cost.backward()
            optimizer.step()
            epoch_cost += cost.item()

        epoch_cost /= len(train_loader)
        accuracy = correct / total * 100

        model3_cost_list.append(epoch_cost)
        model3_accuracy_list.append(accuracy)

        # if epoch % 10 == 0:
        #     print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch, nb_epochs, epoch_cost, accuracy))

    with torch.no_grad():
        correct = 0
        total = 0

        for data, label in tqdm(test_loader):
            label = label.to(device)
            data = data.view(data.shape[0], -1).to(device)
            preds = model3(data)

            preds_label = torch.max(preds.data, 1)[1]
            correct += (preds_label == label).sum().item()
            total += len(label)

        accuracy = correct / total * 100
        print('Test Accuracy: ', accuracy, '%')

    plot_all(nb_epochs, model1_cost_list, model1_accuracy_list, model2_cost_list, model2_accuracy_list, model3_cost_list, model3_accuracy_list)


# def main():
#     train_loader, test_loader = load_mnist()

#     model = model_2().to(device)

#     criterion = nn.CrossEntropyLoss().to(device)
#     optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#     cost_list = []
#     accuracy_list = []
#     for epoch in range(nb_epochs):
#         epoch_cost = 0
#         correct = 0
#         total = 0
#         for data, label in train_loader:
#             optimizer.zero_grad()   # Set gradient to "0"

#             label = label.to(device)
#             data = data.view(data.shape[0], -1).to(device)
#             # data = data.view(data.shape[0], -1)
#             hypothesis = model(data)

#             hypo_label = torch.max(hypothesis.data, 1)[1]
#             correct += (hypo_label == label).sum().item()
#             total += len(label)

#             cost = criterion(hypothesis, label)
#             cost.backward()
#             optimizer.step()
#             epoch_cost += cost.item()

#         epoch_cost /= len(train_loader)
#         accuracy = correct / total * 100

#         cost_list.append(epoch_cost)
#         accuracy_list.append(accuracy)

#         if epoch % 10 == 0:
#             print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch, nb_epochs, epoch_cost, accuracy))

#     plot_one(nb_epochs, cost_list, accuracy_list)

#     with torch.no_grad():
#         correct = 0
#         total = 0

#         for data, label in test_loader:
#             label = label.to(device)
#             data = data.view(data.shape[0], -1).to(device)
#             preds = model(data)

#             preds_label = torch.max(preds.data, 1)[1]
#             correct += (preds_label == label).sum().item()
#             total += len(label)

#         accuracy = correct / total * 100
#         print('Test Accuracy: ', accuracy, '%')


if __name__ == '__main__':
    main()