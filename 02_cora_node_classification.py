import torch
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils    import to_networkx

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv 

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
graph = dataset[0]

print("{} nodes, {} edges".format(graph.num_nodes, graph.num_edges))

## MLP Definition
# Operates directly on a nodes feature set, does not consider the
# graph connectivity at all.
# Note - Notice the 'common' MLP structure here, two linear operations into and 
#        out of a hidden layer, with a nonlinearity applied after the first 
#        linear operation. 
# Note - Drop out is used. I think the training parameter is the mask
#        used to denote the training nodes, so the drop out can only
#        target them. 
class MLP(torch.nn.Module):
	def __init__(self, hidden_channels):
		super(MLP, self).__init__()

		torch.manual_seed(12345)
		self.lin1 = Linear(dataset.num_features, hidden_channels)
		self.lin2 = Linear(hidden_channels, dataset.num_classes)

	def forward(self, x):
		x = self.lin1(x)
		x = x.relu()
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.lin2(x)
		return x

# model = MLP(hidden_channels=16)
# print("Running MLP - No connectivity considered.")
# print(model)
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
	model.train()
	optimizer.zero_grad()
	out  = model(graph.x)
	loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
	loss.backward()
	optimizer.step()
	return loss

def test():
	model.eval()
	out = model(graph.x)
	pred = out.argmax(dim=1) # Actual prediction
	# This returns a new list/tensor composed of entries that are 'correct'
	# in that they match.
	test_correct = pred[graph.test_mask] == graph.y[graph.test_mask]
	test_acc = int(test_correct.sum())/int(graph.test_mask.sum())
	return test_acc

# for epoch in range(1, 201):
# 	loss = train()
# test_acc = test()
# print("final loss: {}, test acc: {}".format(loss, test_acc))

## GNN Definition
class GCN(torch.nn.Module):
	def __init__(self, hidden_channels):
		super(GCN, self).__init__()

		torch.manual_seed(12345)

		self.conv1 = GCNConv(dataset.num_features, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = x.relu()
		x = F.dropout(x, p=0.5, training=self.training)
		x = self.conv2(x, edge_index)
		return x

model = GCN(hidden_channels=16)
print("Running GNN - Now considering connectivity.")
print(model)

# Before training, lets use a TSNE to visualize. 

# model.eval()
# out = model(graph.x, graph.edge_index)
# visualize(out, color=graph.y)

# Now we can train the model to see the improvement.
model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
	model.train()
	optimizer.zero_grad()
	out = model(graph.x, graph.edge_index)
	loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
	loss.backward()
	optimizer.step()
	return loss

def test():
	model.eval()
	out  = model(graph.x, graph.edge_index)
	pred = out.argmax(dim=1)
	test_correct = pred[graph.test_mask] == graph.y[graph.test_mask]
	test_acc = int(test_correct.sum())/int(graph.test_mask.sum()) 
	return test_acc

for epoch in range(1, 201):
	loss = train()

test_acc = test()
print("final loss: {}, test acc: {}".format(loss, test_acc))

out = model(graph.x, graph.edge_index)
visualize(out, color=graph.y)