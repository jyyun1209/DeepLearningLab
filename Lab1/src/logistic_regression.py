from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def split():
	digits = load_digits()
	x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, 
		test_size=0.25, random_state=0)

	train_idx = np.where(y_train < 3)
	y_train = y_train[train_idx]
	x_train = x_train[train_idx]
	train_idx = np.where(y_train > 0)
	y_train = y_train[train_idx]
	x_train = x_train[train_idx]
	
	test_idx = np.where(y_test< 3)
	y_test = y_test[test_idx]
	x_test = x_test[test_idx]
	test_idx = np.where(y_test > 0)
	y_test = y_test[test_idx]
	x_test = x_test[test_idx]
	
	y_train = np.where(y_train == 1, 0, 1)
	y_test = np.where(y_test == 1, 0, 1)
	
	x_train = torch.tensor(x_train, dtype=torch.float)
	x_test = torch.tensor(x_test, dtype=torch.float)
	y_train = torch.tensor(y_train, dtype=torch.float).view(y_train.shape[0], 1)
	y_test = torch.tensor(y_test, dtype=torch.float).view(y_test.shape[0], 1)
	
	return x_train, x_test, y_train, y_test

def plot_cost(nb_epochs, costs):
	costs = np.array(costs)
	
	fig = plt.figure()
	fig.subplots_adjust(top=0.8)
	ax1 = fig.add_subplot(111)
	ax1.set_ylabel('cost')
	ax1.set_xlabel('epochs')
	ax1.set_title('Costs for epochs')
	
	ax1.plot(np.arange(1, nb_epochs + 10, 10), costs)
	plt.show()
	
def main():
	torch.manual_seed(1)
	x_train, x_test, y_train, y_test = split()
	
	model = nn.Sequential(
			nn.Linear(64, 1),
			nn.Sigmoid()
			)
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
	
	costs = []
	nb_epochs = 1000
	for epoch in range(nb_epochs + 1):
		hypothesis = model(x_train)
		cost = F.binary_cross_entropy(hypothesis, y_train)
		if epoch % 10 == 0:
			costs.append(cost.item())
		
		optimizer.zero_grad()
		cost.backward()
		optimizer.step()
		
		if epoch % 10 == 0:
			prediction = hypothesis >= torch.FloatTensor([0.5])
			correct_prediction = prediction.float() == y_train
			accuracy = correct_prediction.sum().item() / len(correct_prediction)
			print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch, nb_epochs, cost.item(), accuracy * 100))

	with torch.no_grad():
		hypothesis = model(x_test)
		cost = F.binary_cross_entropy(hypothesis, y_test)
		prediction = hypothesis >= torch.FloatTensor([0.5])
		correct_prediction = prediction.float() == y_test
		accuracy = correct_prediction.sum().item() / len(correct_prediction)
		print('Cost: {:.6f} Accuracy {:2.2f}%'.format(cost.item(), accuracy * 100))

	plot_cost(nb_epochs, costs)


if __name__ == '__main__':
	main()

