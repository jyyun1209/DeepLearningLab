import numpy as np
from numpy import loadtxt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm

torch.manual_seed(1)

learning_rate = 1e-3
nb_epochs = 2000

class logistic(nn.Module):
    def __init__(self):
        super(logistic, self).__init__()
        self.hidden1 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.sigmoid(x)

        return x

class neuralnet(nn.Module):
    def __init__(self):
        super(neuralnet, self).__init__()
        self.hidden1 = nn.Linear(8, 12)
        self.hidden2 = nn.Linear(12, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.sigmoid(x)        

        return x


def plot_all(nb_epochs, model1_cost_list, model1_accuracy_list, model2_cost_list, model2_accuracy_list):
    model1_cost_list = np.array(model1_cost_list)
    model2_cost_list = np.array(model2_cost_list)
    model1_accuracy_list = np.array(model1_accuracy_list)
    model2_accuracy_list = np.array(model2_accuracy_list)
    
    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    
    ax1 = fig.add_subplot(121)
    ax1.set_ylabel('cost')
    ax1.set_xlabel('epochs')
    ax1.set_title('Costs for epochs')
    ax1.plot(np.arange(1, nb_epochs, 10), model1_cost_list, 'r', label='Logistic_reg')
    ax1.plot(np.arange(1, nb_epochs, 10), model2_cost_list, 'b', label='Neural_Net')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epochs')
    ax2.set_title('Accuracies for epochs')
    ax2.plot(np.arange(1, nb_epochs, 10), model1_accuracy_list, 'r', label='Logistic_reg')
    ax2.plot(np.arange(1, nb_epochs, 10), model2_accuracy_list, 'b', label='Neural_Net')
    ax2.legend()

    plt.show()


def load_data():
    # load the dataset
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
    # split into input (X) and output (y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]

    X_train = X[:600]
    Y_train = Y[:600]
    X_test = X[600:]
    Y_test = Y[600:]

    X_train = torch.tensor(X_train, dtype=torch.float)
    Y_train = torch.tensor(Y_train, dtype=torch.float).view(Y_train.shape[0], 1)
    X_test = torch.tensor(X_test, dtype=torch.float)
    Y_test = torch.tensor(Y_test, dtype=torch.float).view(Y_test.shape[0], 1)

    return X_train, Y_train, X_test, Y_test


def main():
    X_train, Y_train, X_test, Y_test = load_data()

    ##### Logistic Regression #####
    logistic_reg = logistic()
    optimizer = optim.SGD(logistic_reg.parameters(), lr=learning_rate)

    lr_cost_list = []
    lr_accuracy_list = []
    for epoch in tqdm(range(nb_epochs)):
        correct = 0
        total = 0

        optimizer.zero_grad()   # Set gradient to "0"

        hypothesis = logistic_reg(X_train)

        hypo_label = hypothesis >= torch.FloatTensor([0.5])
        correct += (hypo_label == Y_train).sum().item()
        total += len(Y_train)

        cost = F.binary_cross_entropy(hypothesis, Y_train)
        cost.backward()
        optimizer.step()

        correct_prediction = hypo_label.float() == Y_train
        accuracy = (correct_prediction.sum().item() / len(correct_prediction)) * 100

        if epoch % 10 == 0:
            lr_cost_list.append(cost.item())
            lr_accuracy_list.append(accuracy)

    with torch.no_grad():
        preds = logistic_reg(X_test)
        preds_label = preds >= torch.FloatTensor([0.5])
        correct_prediction = preds_label.float() == Y_test
        accuracy = (correct_prediction.sum().item() / len(correct_prediction)) * 100
        print('Test Accuracy: ', accuracy, '%')

    ##### Neural Network #####
    neural_net = neuralnet()
    optimizer = optim.SGD(neural_net.parameters(), lr=learning_rate)

    nn_cost_list = []
    nn_accuracy_list = []
    for epoch in tqdm(range(nb_epochs)):
        correct = 0
        total = 0

        optimizer.zero_grad()   # Set gradient to "0"

        hypothesis = neural_net(X_train)

        hypo_label = hypothesis >= torch.FloatTensor([0.5])
        correct += (hypo_label == Y_train).sum().item()
        total += len(Y_train)

        cost = F.binary_cross_entropy(hypothesis, Y_train)
        cost.backward()
        optimizer.step()

        correct_prediction = hypo_label.float() == Y_train
        accuracy = (correct_prediction.sum().item() / len(correct_prediction)) * 100

        if epoch % 10 == 0:
            nn_cost_list.append(cost.item())
            nn_accuracy_list.append(accuracy)
    

    with torch.no_grad():
        preds = neural_net(X_test)
        preds_label = preds >= torch.FloatTensor([0.5])
        correct_prediction = preds_label.float() == Y_test
        accuracy = (correct_prediction.sum().item() / len(correct_prediction)) * 100
        print('Test Accuracy: ', accuracy, '%')

    plot_all(nb_epochs, lr_cost_list, lr_accuracy_list, nn_cost_list, nn_accuracy_list)

if __name__ == '__main__':
    main()