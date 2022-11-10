import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# nb_epochs = 90

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


def plot_all(nb_epochs, model1_cost_list, model1_accuracy_list, model2_cost_list, model2_accuracy_list):
    model1_cost_list = np.array(model1_cost_list)
    model2_cost_list = np.array(model2_cost_list)
    model1_accuracy_list = np.array(model1_accuracy_list)
    model2_accuracy_list = np.array(model2_accuracy_list)
    
    fig = plt.figure()
    fig.subplots_adjust(top=0.8)
    
    ax1 = fig.add_subplot(121)
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epochs')
    ax1.set_title('Losses for epochs')
    ax1.plot(np.arange(1, nb_epochs + 1, 1), model1_cost_list, 'r', label='Train')
    ax1.plot(np.arange(1, nb_epochs + 1, 1), model2_cost_list, 'b', label='Val')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('epochs')
    ax2.set_title('Accuracies for epochs')
    ax2.plot(np.arange(1, nb_epochs + 1, 1), model1_accuracy_list, 'r', label='Train')
    ax2.plot(np.arange(1, nb_epochs + 1, 1), model2_accuracy_list, 'b', label='Val')
    ax2.legend()

    plt.show()


def readpkl(filepath):
    with open(filepath, "rb") as f:
        lst = pickle.load(f)

    return lst

def main():
    train_loss_list = readpkl('resume_from_pretrained_model/train_loss_list.pkl')
    train_acc_list = readpkl('resume_from_pretrained_model/train_acc_list.pkl')
    val_loss_list = readpkl('resume_from_pretrained_model/val_loss_list.pkl')
    val_acc_list = readpkl('resume_from_pretrained_model/val_acc_list.pkl')

    train_loss_list = np.array(train_loss_list)
    train_acc_list = np.array(train_acc_list)
    val_loss_list = np.array(val_loss_list)
    val_acc_list = np.array(val_acc_list)

    plot_all(len(train_loss_list), train_loss_list, train_acc_list, val_loss_list, val_acc_list)


if __name__ == '__main__':
    main()