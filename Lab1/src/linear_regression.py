import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import R2Score

def split(csv_reader):
	x_train = []
	y_train = []
	x_test = []
	y_test = []
	for i, row in enumerate(csv_reader):
		if i == 0:
			continue
		if i <= 300:
			x_train.append(np.array(row[1:-1], dtype='float'))
			y_train.append(np.array([row[-1]], dtype='float'))
		else:
			x_test.append(np.array(row[1:-1], dtype='float'))
			y_test.append(np.array([row[-1]], dtype='float'))

	x_train = torch.tensor(np.stack(x_train), dtype=torch.float)
	y_train = torch.tensor(np.stack(y_train), dtype=torch.float)
	x_test = torch.tensor(np.stack(x_test), dtype=torch.float)
	y_test = torch.tensor(np.stack(y_test), dtype=torch.float)

	return x_train, y_train, x_test, y_test

def main():
	torch.manual_seed(1)

	with open('/share/DLL/Lab1/Real_estate.csv', newline='') as csvfile:
		csv_reader = csv.reader(csvfile, delimiter=',')
		x_train, y_train, x_test, y_test = split(csv_reader)	
		model = nn.Linear(6, 1)
#		optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
		optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

		nb_epochs = 2000
		for epoch in range(nb_epochs):
			prediction = model(x_train)
			cost = F.mse_loss(prediction, y_train)
			optimizer.zero_grad()
			cost.backward()
			optimizer.step()

			if epoch % 100 == 0:
				print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

		with torch.no_grad():
			prediction = model(x_test)
			cost = F.mse_loss(prediction, y_test)
			print('\nCost: ', cost.item())
			r2score = R2Score(num_outputs=1)
			r2 = r2score(prediction, y_test)
			print('R2 score from Pytorch: ', r2.item())
			
if __name__ == '__main__':
	main()
