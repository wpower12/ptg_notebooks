import time
import torch
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.datasets import KarateClub
from torch_geometric.utils    import to_networkx

from torch.nn import Linear
from torch_geometric.nn import GCNConv 

# Tidy. 
def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()


## Getting some data 
dataset = KarateClub()
graph = dataset[0]
G_nx  = to_networkx(graph, to_undirected=True)

# Uncomment for initial vis. 

# visualize(G_nx, color=graph.y)

## Defining a GCN Model
# The meat!!!
class GCN(torch.nn.Module):
	def __init__(self):
		super(GCN, self).__init__()
		torch.manual_seed(12345)

		# set up the building blocks
		# 3 GCNs means we look at a 3 hop neighborhood
		self.conv1 = GCNConv(dataset.num_features, 4)
		self.conv2 = GCNConv(4, 4)
		self.conv3 = GCNConv(4, 2)
		# Final classifing layer
		self.classifier = Linear(2, dataset.num_classes)

	def forward(self, x, edge_index):
		# The three conv layers need the edge_index. 
		h = self.conv1(x, edge_index) # all nodes have 1-hop data now
		h = h.tanh()
		h = self.conv2(h, edge_index) # now they have 2-hop data
		h = h.tanh() 
		h = self.conv3(h, edge_index) # and finally, they contain 3-hop data
		h = h.tanh()

		# Note the final layer doesnt need connectivity data, its just
		# using the final node embedding to make the classification.
		out = self.classifier(h)

		return out, h

model = GCN()
print(model)

## Initial Embedding Vis
# This will visualize the initial, random embeddings of the nodes
# highlighting the strong inductive bias of the model family. Nodes are
# already pretty well grouped just on the basis of the raw edge data.
# 
# uncomment for initial vis

# _, h = model(graph.x, graph.edge_index)
# visualize(h, color=graph.y)

## Training Better Embeddings
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(data):
	optimizer.zero_grad()
	out, h = model(data.x, data.edge_index)
	loss = criterion(out[data.train_mask], data.y[data.train_mask])
	loss.backward()
	optimizer.step()
	return loss, h

for epoch in range(401):
	loss, h = train(graph)


# loss, h = train(graph)
visualize(h, color=graph.y, epoch=epoch, loss=loss)