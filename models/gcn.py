import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

####读取数据
y_train1 = np.load('train_label.npy')
x_train = np.load('r18_train_feature.npy')
y_test1 = np.load('test_label.npy')
x_test = np.load('r18_test_feature.npy')

#建立邻接矩阵
from scipy.spatial import distance
distv = distance.pdist(x_train, metric='correlation')
dist0 = distance.squareform(distv)
sigma = np.mean(dist0)
sparse_graph1 = np.exp(- dist0 ** 2 / (2 * sigma ** 2))
adj =  sparse_graph1

adj_train = torch.FloatTensor(adj)
edge_index_train = adj_train

distv = distance.pdist(x_test, metric='correlation')
dist0 = distance.squareform(distv)
sigma = np.mean(dist0)
sparse_graph1 = np.exp(- dist0 ** 2 / (2 * sigma ** 2))
adj =  sparse_graph1

adj_test = torch.FloatTensor(adj)
edge_index_test = adj_test
#################

x_train = torch.tensor(x_train, dtype=torch.float)
x_test = torch.tensor(x_test, dtype=torch.float)
y_train1 = torch.LongTensor(y_train1).squeeze()
y_test1 = torch.LongTensor(y_test1).squeeze()

#################

#图卷积网络
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        return output


    
class GCN(torch.nn.Module):
    def __init__(self, input_channles,hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(123456)
        self.conv1 = GraphConvolution(input_channles, hidden_channels)
        self.conv2 = GraphConvolution(hidden_channels, 3)
       # self.norm1 = nn.LayerNorm(16)
       # self.norm2 = nn.LayerNorm(3)
        
        
        
    def forward(self, x, adj):
        
        x = self.conv1(x, adj)
       # x = self.norm1(x)
        x = x.relu()
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, adj)
      #  x = self.norm2(x)

        return x
#######################
#训练模型

model = GCN(512,16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()  
    out = model(x_train, edge_index_train)  
    loss = criterion(out, y_train1)  
    loss.backward()  
    optimizer.step()  
    return loss

def test():
    model.eval()
    out = model(x_test, edge_index_test)
    pred = out.argmax(dim=1)  

    test_correct = pred == y_test1  
    test_acc = int(test_correct.sum()) / int(y_test1.shape[0])  
    return test_acc,pred



for epoch in range(1, 500):
    loss = train()
    test_acc,pred = test()        
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, test_acc: {test_acc:.5f}')
